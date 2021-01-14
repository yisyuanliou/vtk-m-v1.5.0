#ifndef vtk_m_worklet_GMMTraining_h
#define vtk_m_worklet_GMMTraining_h

#include <vtkm/Math.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/worklet/FieldHistogram.h>
#include <vtkm/Matrix.h>
#include <vtkm/cont/Timer.h>

#include <vtkm/cont/Field.h>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>

#include <vtkm/Eigen/Dense>
using namespace Eigen;

namespace vtkm {
namespace worklet {

//#define OMP_THREAD_NUM 28 //allow openmp threads

template <int GMs, int VARs, typename DType>
struct GMM {
    VTKM_EXEC_CONT
    GMM(){}

    //Set a GMM weigths means and covariance matrixs
    //use case: train gmm before and store them in files. Then load the files later to use the GMMs (without training again)
    //paramaters: 
    // w (float array): weights of a GMM (length is the same as number of Gaussian components) (from Gau component 0 to GMs-1 )
    // m (float array): means of a GMM (length is the "number of Gaussian components" * "VARs") (from Gau component 0 to GMs-1 )
    // cov(float array): covariance matrixs of a GMM (length is the "number of Gaussian components" * "number of elements of full matrix") (from Gau component 0 to GMs-1 )
    //                   use 3x3 matrix as an example first (assume m20 is the element at row:2 and col:0)
    //                   the order to pass cov should be m00 m01 m02 m10 m11 m12 m20 m21 m22 (row major)
    VTKM_EXEC_CONT
    void setGMM(DType *w, DType*m, DType* cov)
    {
        for( int c=0; c<GMs; c++ ){
            weights[c] = w[c];
            for( int i=0; i<VARs; i++ ){
                means[c][i] = m[c * VARs + i];

            }
        
            //set covariance matrix, ordered by row major
            int counter = 0;
            for( int ro=0; ro<VARs; ro++ ){
                for( int co=0; co<VARs; co++ ){
                    covMats[c][ro][co] = cov[ c*VARs*VARs + counter ];
                    counter ++;
                }
            }
        }
    }

    VTKM_EXEC_CONT
    vtkm::Vec<DType, VARs> getSample( const vtkm::Vec<DType, VARs>& rP3D, const DType &r ) {
        //pickup a component
        vtkm::Int32 c;
        DType sum = 0.0f;
        for( c=0; c<GMs; c++ ){
            sum += weights[c];
            if( sum > r )break;
        }
        if( c == GMs )c = GMs - 1;

        //transform the above sample by the covariance matrix (sP3D = eigenMat * rP3D)
        vtkm::Vec<DType, VARs> sP3D;
        for( int i=0; i<VARs; i++ ){
            sP3D[i] = 0;
            for( int j=0; j<VARs; j++ ){
                sP3D[i] += lowerMat[c][i][j]*rP3D[j];    
            }
            sP3D[i] += means[c][i];
        }
        // if( sP3D[0] - means[c][0] > 3*sqrt(covMats[c][0][0]) ) sP3D[0] = means[c][0] + 3*sqrt(covMats[c][0][0]);
        // else if( means[c][0] - sP3D[0] > 3*sqrt(covMats[c][0][0]) ) sP3D[0] = means[c][0] - 3*sqrt(covMats[c][0][0]);
        //if(sP3D[0]<1.5)sP3D[0]=1.5;



//       if( sP3D[0] < 0 ){
  //          printf("v %f   %f    %f    %f\n", sP3D[0], lowerMat[0][0][0], means[0][0], rP3D[0]);
    //        exit(0);
     //   }

        return sP3D;
    }

    VTKM_EXEC_CONT
    DType getLogPdfFromOneComponent(const vtkm::Vec<DType, VARs> &p, vtkm::Int32 c)
    {
        // x = p - mean
        DType x[VARs];
        for( int i=0; i<VARs; i++ )x[i] = p[i] - means[c][i];

        // tmp = x * uMat
        DType tmp[VARs];
        for( int j=0; j<VARs; j++ ){
            tmp[j] = 0;
            for( int k=0; k<VARs; k++ ){
                tmp[j] += x[k]*precCholMat[c][k][j];
            }
        }

        // density = inner_Product(tmp, tmp);
        DType density = 0;
        for( int k=0; k<VARs; k++ ){
            density += tmp[k]*tmp[k];
        }

        return -0.5 * (VARs * 1.83787706641 + density ) + logPDet[c];
    }

    VTKM_EXEC_CONT
    DType getCompWgtLogProbability(const vtkm::Vec<DType, VARs> &p, vtkm::Int32 c)
    {
        return log(weights[c]) + getLogPdfFromOneComponent(p, c);
    }

    VTKM_EXEC_CONT
    DType getProbability(const vtkm::Vec<DType, VARs> &p)
    {
        DType prob = 0.0f;
        for( int i=0; i<GMs; i++ ){
            prob += exp ( log(weights[i]) + getLogPdfFromOneComponent(p, i) );
        }

        return prob;
    }

    DType weights[GMs];
    DType means[GMs][VARs];
    DType covMats[GMs][VARs][VARs];
    DType logPDet[GMs];
    DType precCholMat[GMs][VARs][VARs];
    DType lowerMat[GMs][VARs][VARs];
    //DType det[GMs];
    //DType inverseMat[GMs][VARs][VARs];
};

//simple functor that returns basic statistics
template <int GMs, int VARs, typename DType>
class GMMTraining
{
public:
    vtkm::cont::ArrayHandle< GMM<GMs, VARs, DType> > gmmsHandle;

    // E Step
    class EStep : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut, FieldOut, WholeArrayIn, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6);


        VTKM_CONT
        EStep() {}

        template<typename NonstopWholeArrayInPortalType, typename GMMsWholeArrayInPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs> &dataPoint, const vtkm::Int32 &gmmId,
                        vtkm::Vec<DType, GMs> &resposibilityOut,
                        DType &logProbNorm, 
                        NonstopWholeArrayInPortalType& nonstopWholeArrayInPortal,
                        GMMsWholeArrayInPortalType& gmmsWholeArrayInPortal ) const
        {
            if( nonstopWholeArrayInPortal.Get(gmmId) == 0 ){ 
                logProbNorm = std::numeric_limits<DType>::epsilon();
                for( int c=0; c<GMs; c++ ){
		    resposibilityOut[c] = std::numeric_limits<DType>::epsilon();
                }
            }else{
                GMM<GMs, VARs, DType> gmm = gmmsWholeArrayInPortal.Get(gmmId);

                vtkm::Vec<DType, GMs> score;

                // Compute Score from a dataPoint to each component and get max score
                logProbNorm = 0;
                for( int c = 0; c<GMs; c++ ){
                    score[c] = gmm.getCompWgtLogProbability(dataPoint, c);
                    //  std::numeric_limits<DType>::min() (if DType is float or double), this is a smallest positive value
                    // is the following statement correct?
                    // but, vtkm::Exp(score[c]) should be >= 0 (it is probability) , so this may be ok 
                    // check what really we want here.***********
                    if( vtkm::Exp(score[c]) < std::numeric_limits<DType>::min() ) {
                        score[c] = vtkm::Log(std::numeric_limits<DType>::min());
                    }
                    logProbNorm += vtkm::Exp( score[c] );
                }
                logProbNorm = log( logProbNorm );
                for( int c=0; c<GMs; c++ ){
                    resposibilityOut[c] = vtkm::Exp( score[c] - logProbNorm );
                }
            }
        }
    };

    // Divide an array by a scalar
    class DivideArrayByArray : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2, _3);


        VTKM_CONT
        DivideArrayByArray() {}

        template<typename T, typename U>
        VTKM_EXEC
        void operator()(const T &a, const U &b,
                        T &c ) const
        {
            c = a / static_cast<DType>(b);
        }
    };

    // Divide an array by a scalar
    class CheckStop : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        DType StopThreshold;

        VTKM_CONT
        CheckStop( DType _stopThreshold ) :StopThreshold(_stopThreshold) {}

        template<typename T>
        VTKM_EXEC
        void operator()(const T &a, const T &b, const vtkm::Int32 &prevNonStopFlag, vtkm::Int32 &nonStopFlag ) const
        {
            if( prevNonStopFlag == 0 )nonStopFlag = 0;
            else{
                DType diff = vtkm::Abs(a-b);
                if( diff > StopThreshold ) nonStopFlag = 1;
                else nonStopFlag = 0;
            }
        }
    };

    class UpdateWeight : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn, WholeArrayOut);
        typedef void ExecutionSignature(_1, _2, _3, _4);


        VTKM_CONT
        UpdateWeight() {}

        template<typename GMMsWholeArrayOutPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, GMs> &weights, const vtkm::Int32 &gmmId, 
                        const vtkm::Int32 &nonstop,
                        GMMsWholeArrayOutPortalType &gmmsOutPortal ) const
		{
            if( nonstop == 1){
                GMM<GMs, VARs, DType> gmm = gmmsOutPortal.Get(gmmId);
                //Calculate Sum of weights
                DType sumWeight = 0.0f;
                for( int i=0; i<GMs; i++ ) sumWeight += (weights[i] + 10 * std::numeric_limits<DType>::epsilon() );
                for( int i=0; i<GMs; i++ )            
                    gmm.weights[i] = (weights[i] + 10 * std::numeric_limits<DType>::epsilon()) / sumWeight;
                gmmsOutPortal.Set(gmmId, gmm);
            }
        }
    };

    // Upadate Mean
    class ComputeUpdatedMean : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldOut, 
                                      WholeArrayIn, WholeArrayIn, 
                                      WholeArrayIn, WholeArrayIn,
                                      WholeArrayIn );
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

        VTKM_CONT
        ComputeUpdatedMean() {}

        template<typename TrainDataPortalType, 
                 typename GmmTrainSampleSncPortalType, 
                 typename RawWeightPortalType,
                 typename ResponsibilityPortalType,
                 typename NonStopPortalType >
        VTKM_EXEC
        void operator()(const vtkm::Vec< vtkm::Int32, 3 > &meanEltIdx,
                        DType &out,
                        TrainDataPortalType &trainDataPortal,
                        GmmTrainSampleSncPortalType gmmTrainSampleSncPortal,
                        RawWeightPortalType rawWeightPortal,
                        ResponsibilityPortalType responsibilityPortal,
                        NonStopPortalType nonStopPortal ) const
        {
            int gmmId = meanEltIdx[0];
            int comp = meanEltIdx[1];
            int var = meanEltIdx[2];
            if( nonStopPortal.Get( gmmId ) == 1 ){
                int trainDataStart = gmmTrainSampleSncPortal.Get(gmmId)[0];
                int trainDataCount = gmmTrainSampleSncPortal.Get(gmmId)[1];

                out = 0.0f;
                for( int s = trainDataStart; s < trainDataStart + trainDataCount; s++ )
                {
                    out += responsibilityPortal.Get(s)[comp] * trainDataPortal.Get(s)[var];       
                }
                out /= ( rawWeightPortal.Get(gmmId)[comp] + 10.0f * std::numeric_limits<DType>::epsilon() );
            }
        }
    };

    // Upadate Covariance Matrix
    class UpdateMeanOneShot : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, WholeArrayIn, WholeArrayIn, WholeArrayOut);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        VTKM_CONT
        UpdateMeanOneShot() {}

        template<typename MeanUpdataPortalType, typename NonStopPortalType,
                typename GMMsWholeArrayOutPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &gmmId,
                        MeanUpdataPortalType &meanUpdatePortal,
                        NonStopPortalType nonStopPortal, 
                        GMMsWholeArrayOutPortalType &gmmsOutPortal) const
        {
            if( nonStopPortal.Get( gmmId ) == 1 ){
                GMM<GMs, VARs, DType> gmm = gmmsOutPortal.Get(gmmId);

                int stMeanIdx = gmmId * GMs * VARs;
                for( int comp=0; comp<GMs; comp++ )
                {
                    for( int v=0; v<VARs; v++ )
                    {
                        gmm.means[comp][v] = meanUpdatePortal.Get(stMeanIdx);
                        stMeanIdx ++;
                    }
                }
                gmmsOutPortal.Set(gmmId, gmm);
            }
        }
    };

    // Upadate Covariance Maatrix
    class ComputeUpdatedCovMatrix : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldOut, 
                                      WholeArrayIn, WholeArrayIn, 
                                      WholeArrayIn, WholeArrayIn,
                                      WholeArrayIn, WholeArrayIn );
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);

        DType minCovDiag;

        VTKM_CONT
        ComputeUpdatedCovMatrix(DType _minCovDiag) :minCovDiag(_minCovDiag){}

        template<typename TrainDataPortalType, 
                 typename GmmTrainSampleSncPortalType, typename GmmsHandlePortalType,
                 typename ResponsibilityPortalType, typename RawWeightPortalType, 
                 typename NonStopPortalType >
        VTKM_EXEC
        void operator()(const vtkm::Vec< vtkm::Int32, 4 > &covEltIdx,
                        DType &out,
                        TrainDataPortalType &trainDataPortal,
                        GmmTrainSampleSncPortalType gmmTrainSampleSncPortal,
                        GmmsHandlePortalType gmmHandlePortal,
                        ResponsibilityPortalType responsibilityPortal,
                        RawWeightPortalType rawWeightPortal,
                        NonStopPortalType nonStopPortal ) const
        {
            int gmmId = covEltIdx[0];
            int comp = covEltIdx[1];
            int row = covEltIdx[2];
            int col = covEltIdx[3];
            if( nonStopPortal.Get( gmmId ) == 1 ){
                int trainDataStart = gmmTrainSampleSncPortal.Get(gmmId)[0];
                int trainDataCount = gmmTrainSampleSncPortal.Get(gmmId)[1];

                GMM<GMs, VARs, DType> gmm = gmmHandlePortal.Get(gmmId);

                out = 0.0f;
                for( int s = trainDataStart; s < trainDataStart + trainDataCount; s++ )
                {
                    out += responsibilityPortal.Get(s)[comp]  
                           * ( trainDataPortal.Get(s)[row] - gmm.means[comp][row] )
                           * ( trainDataPortal.Get(s)[col] - gmm.means[comp][col] );
                }
                out /= ( rawWeightPortal.Get(gmmId)[comp] + 10.0f * std::numeric_limits<DType>::epsilon() );
                if( row == col ) out += minCovDiag;
            }
        }
    };

    // Upadate Covariance Matrix
    class UpdateCovMatrixOneShot : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, WholeArrayIn, WholeArrayIn, WholeArrayOut,
                                      WholeArrayIn, WholeArrayIn );
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6);

        vtkm::Int32 MatTriangleSize;

        VTKM_CONT
        UpdateCovMatrixOneShot(vtkm::Int32 _MatTriangleSize) :MatTriangleSize(_MatTriangleSize) {}

        template<typename CovUpdataPortalType, typename NonStopPortalType,
                typename GMMsWholeArrayOutPortalType, typename Idx2RowPortalType, 
                typename Idx2ColPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &gmmId,
                        CovUpdataPortalType &covUpdatePortal,
                        NonStopPortalType nonStopPortal, 
                        GMMsWholeArrayOutPortalType &gmmsOutPortal,
                        Idx2RowPortalType& idx2RowPortal, 
                        Idx2ColPortalType& idx2ColPortal) const
        {
            if( nonStopPortal.Get( gmmId ) == 1 ){
                GMM<GMs, VARs, DType> gmm = gmmsOutPortal.Get(gmmId);

                int stGMMCovIdx = gmmId * GMs * MatTriangleSize;
                for( int comp=0; comp<GMs; comp++ )
                {
                    int stGauCovIdx = stGMMCovIdx + comp * MatTriangleSize;
                    for( int i=0; i< MatTriangleSize; i++ )
                    {
                        int rw = idx2RowPortal.Get(i);
                        int cl = idx2ColPortal.Get(i);
                        gmm.covMats[comp][rw][cl] = covUpdatePortal.Get(stGauCovIdx + i);
                        if( rw != cl ) gmm.covMats[comp][cl][rw] = gmm.covMats[comp][rw][cl];
                    }
                }
                gmmsOutPortal.Set(gmmId, gmm);
            }
        }
    };

    // Sampling
    class Sampling : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldOut, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4, _5);

        VTKM_CONT
        Sampling() {}

        template<typename GMMsWholeArrayInPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &gmmId, const vtkm::Vec<DType, VARs> &rNormal3D, const DType &rUniform, 
                        vtkm::Vec<DType, VARs> &sample, GMMsWholeArrayInPortalType &gmmsInPortal ) const
        {
            //GMM<GMs, VARs> gmm = gmmsInPortal.Get(gmmId);
            //sample = gmm.getSample();
            sample = gmmsInPortal.Get(gmmId).getSample(rNormal3D, rUniform);
        }
    };

    // Combine to vector to one
    class CombineTwoArrays : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2, _3);


        VTKM_CONT
        CombineTwoArrays() {}

        template<typename T>
        VTKM_EXEC
        void operator()(const T &a, const T &b,
                        vtkm::Vec<T, 2> &c ) const
        {
            c[0] = a;
            c[1] = b;
        }
    };

    // Mean indicator
    class ComputeMeanIndicator : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2);

        vtkm::Int32 NGMMs;

        VTKM_CONT
        ComputeMeanIndicator( vtkm::Int32 _NGmms ) :NGMMs(_NGmms) {}

        VTKM_EXEC
        void operator()(const vtkm::Int32 &idx, vtkm::Vec<vtkm::Int32, 3> &out ) const
        {
            int tmp = idx;
            out[2] = tmp % ( VARs );
            tmp = ( tmp - out[2] ) / VARs;
            out[1] = tmp % GMs;
            tmp = ( tmp - out[1] ) / GMs;
            out[0] = tmp;
        }
    };

    // Compute Covariance matrix indicator
    class ComputeCovMatIndicator : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldOut, WholeArrayIn, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        vtkm::Int32 MatTriangleSize;

        VTKM_CONT
        ComputeCovMatIndicator( vtkm::Int32 _MatTriangleSize ) :MatTriangleSize(_MatTriangleSize) 
        {}

        template<typename Idx2RowPortalType, typename Idx2ColPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &idx, vtkm::Vec<vtkm::Int32, 4> &out,
                        Idx2RowPortalType& idx2RowPortal, Idx2ColPortalType& idx2ColPortal ) const
        {
            int tmp = idx;
            int uTriIdx = tmp % MatTriangleSize;
            out[3] = idx2ColPortal.Get(uTriIdx);
            out[2] = idx2RowPortal.Get(uTriIdx);
            tmp = (tmp - uTriIdx) / MatTriangleSize;
            out[1] = tmp % GMs;
            tmp = ( tmp - out[1] ) / GMs;
            out[0] = tmp;
        }
    };

  //Run EM algorithm for GMMs training
  //parameters:
  //trainData(std::vector< vtkm::worklet::vtkm::Vec<DType, VARs> >) : training samples
  //                              std::vector is whole training samples which will be trainned for all GMMs
  //                              vtkm::worklet::vtkm::Vec<DType, VARs> is a N-variable training sample
  //gmmIds(std::vector<vtkm::Int32>) : a sample is correspoding to which GMM
  //                              each elements in this vector is a id(integer)
  //                              trainData and gmmIds must have the same length
  //                              each value in this vector indicates the correspoding sample in trainData's gmmId
  //                              (GMM ID should start from 0 and increase)
  //                              The samples in trainData which will generate a GMM should have the same gmmId
  //numGmms(int): total number of GMMs
  //maxInterations(int): maximal number of EM iteration
  //stopTol: stop threshold, if abs(pre_likelihood - current_likelihood)<stopTol, this gmm stop traning
  //verbose: 0:not terminal output, 1:only interations, 2:all step details 
  //clusterLabel: initial conditional for GMM training, initial that a training sample belongs to which Gaussian
  //              the length is the number of training samples, value is between 0 - numberGauComps
  //device: run on which device (GPU CPU or TBB)
  //initState:  0: use random to initalize(clusterLabel is useless), 1: use clusterLabel to do initialize
  void Run( std::vector<vtkm::Vec<DType, VARs>> trainData, std::vector<vtkm::Int32> gmmIds, vtkm::Int32 numGmms, 
            vtkm::Int32 maxInterations, std::vector<vtkm::Int32> clusterLabel, vtkm::Int32 initState = 0, vtkm::Int32 verbose = 0, vtkm::Float32 stopTol = 0.001, DType minCovDiag = 0.000001 )
  {

    //omp_set_num_threads(OMP_THREAD_NUM);
    vtkm::Int32 nTrainData = trainData.size();
    vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>> trainDataHandle = vtkm::cont::make_ArrayHandle(trainData);
    vtkm::cont::ArrayHandle<vtkm::Int32> gmmIdsHandle = vtkm::cont::make_ArrayHandle(gmmIds);
    gmmsHandle.Allocate(numGmms);

    // compute start train data index of each GMM in training Data
    vtkm::cont::ArrayHandle<vtkm::Int32> begGmmIdxInData;
    {
        vtkm::cont::ArrayHandleCounting<vtkm::Int32> coutingConstHandle(0, 1, trainDataHandle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<vtkm::Int32> countingHandle;
		vtkm::cont::Algorithm::Copy(coutingConstHandle, countingHandle);
        vtkm::cont::ArrayHandle<vtkm::Int32> keyOut;
		vtkm::cont::Algorithm::ReduceByKey(gmmIdsHandle, countingHandle, keyOut, begGmmIdxInData, vtkm::Minimum());
        //PrintArray(begGmmIdxInData);
    }
    // Compute number of train data in each GMM
    vtkm::cont::ArrayHandle<vtkm::Int32> numDataInGMM;
    {
        vtkm::cont::ArrayHandleConstant<vtkm::Int32> onesConstHandle(1, trainDataHandle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<vtkm::Int32> onesHandle;
		vtkm::cont::Algorithm::Copy(onesConstHandle, onesHandle);
        vtkm::cont::ArrayHandle<vtkm::Int32> keyOut;
		vtkm::cont::Algorithm::ReduceByKey(gmmIdsHandle, onesHandle, keyOut, numDataInGMM, vtkm::Add());
        //PrintArray(numDataInGMM);
    }
	
    // EM interations
    vtkm::cont::ArrayHandle<DType> current_log_prob_norm;
    vtkm::cont::ArrayHandleConstant<DType> nansConstHandle(NAN, numGmms);
	vtkm::cont::Algorithm::Copy(nansConstHandle, current_log_prob_norm);
    vtkm::cont::ArrayHandle<DType> log_prob_norms;
    log_prob_norms.Allocate(nTrainData);
    vtkm::cont::ArrayHandle< vtkm::Vec<DType, GMs> > resposibilities;
    resposibilities.Allocate(nTrainData);
    vtkm::cont::ArrayHandle<DType> prev_log_prob_norm;
    vtkm::cont::ArrayHandle< vtkm::Vec<DType, GMs> > weights;
    vtkm::cont::ArrayHandle<vtkm::Int32> keyOut_current_log_prob_norm;
    vtkm::cont::ArrayHandle<vtkm::Int32> keyOut_weights;
    vtkm::cont::ArrayHandle< DType > newMeans;
    vtkm::cont::ArrayHandle< DType > newCovarianceMats;
    vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 2> > gmmTrainSampleSnc;
    vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 3> > meanUpdateIndicator;
    vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 4> > covUpdateIndicator;
    vtkm::cont::ArrayHandle<vtkm::Int32> nonstopFlag; 

    //gmmTrainSample snc and mean cov mat indicator computation
    vtkm::worklet::DispatcherMapField<CombineTwoArrays> combineTwoArraysDispatcher( CombineTwoArrays{} );
    combineTwoArraysDispatcher.Invoke( begGmmIdxInData, numDataInGMM, gmmTrainSampleSnc);

    ComputeMeanIndicator computeMeanIndicator(numGmms);
    vtkm::worklet::DispatcherMapField<ComputeMeanIndicator> computeMeanIndicatorDispatcher( computeMeanIndicator );
    computeMeanIndicatorDispatcher.Invoke(vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numGmms*GMs*VARs), 
                                            meanUpdateIndicator );

    //Compute idx to row and col in a upper triangle matrix
    vtkm::Int32 matTriangleSize = ((1+VARs)*VARs)/2;
    std::vector<vtkm::Int32> idx2Row;
    std::vector<vtkm::Int32> idx2Col;
    for( int rw=0; rw<VARs; rw++ ){
        for( int cl=0; cl<rw+1; cl++ ){
            idx2Row.push_back(rw);
            idx2Col.push_back(cl);
        }
    }
    vtkm::cont::ArrayHandle<vtkm::Int32> idx2RowHandle = vtkm::cont::make_ArrayHandle(idx2Row);
    vtkm::cont::ArrayHandle<vtkm::Int32> idx2ColHandle = vtkm::cont::make_ArrayHandle(idx2Col);
    ComputeCovMatIndicator computeCovMatIndicator(matTriangleSize);
    vtkm::worklet::DispatcherMapField<ComputeCovMatIndicator> computeCovMatIndicatorDispatcher( computeCovMatIndicator );
    computeCovMatIndicatorDispatcher.Invoke(vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numGmms*GMs*matTriangleSize ), covUpdateIndicator, idx2RowHandle, idx2ColHandle ); //Only Upper triangle

    //init nonstop flag (all non-stop in the begining)
    vtkm::cont::ArrayHandleConstant<vtkm::Int32> initNonstopFlagConstHandle(1, numGmms);
	vtkm::cont::Algorithm::Copy( initNonstopFlagConstHandle, nonstopFlag );

    if( initState == 0 ){
        //Initialize parameter (random the resposibility matrix)
        for( int i=0; i<nTrainData; i++ ){ //Serial part
            vtkm::Vec<DType, GMs> resTmp;
            DType sum = 0;
            for(int j=0; j<GMs; j++ ){
                resTmp[j]= rand();
                sum += resTmp[j];
            }
            for( int j=0; j<GMs; j++ ) resTmp[j] /= sum;
            resposibilities.GetPortalControl().Set(i, resTmp);
        }
    }
    else{//initState == 1
        //Initialize parameter (from given cluster label)
        for( int i=0; i<nTrainData; i++ ){ //Serial part (modify to parallel later)
            vtkm::Vec<DType, GMs> resTmp;
            for( int j=0; j<GMs; j++ ){
                if( clusterLabel[i] == j ) resTmp[j] = 1;
                else resTmp[j] = 0;
            }
            resposibilities.GetPortalControl().Set(i, resTmp);
        }
    }
    ParameterEstimation(numGmms, gmmIdsHandle, resposibilities, keyOut_weights, weights, nonstopFlag,
                            gmmTrainSampleSnc, meanUpdateIndicator, covUpdateIndicator, newMeans, newCovarianceMats,
                            trainDataHandle, idx2RowHandle, idx2ColHandle, matTriangleSize, 
                            verbose, minCovDiag); //estimate init parameters by random resposibility
	
    // Main EM interations
    for( vtkm::Int32 iter = 0; iter<maxInterations; iter++ )
    {   
		//std::cout << "EM: " << iter << std::endl;
        //if( verbose >=1 ) std::cout << "Iterations: " << iter << std::endl;
		vtkm::cont::Algorithm::Copy( current_log_prob_norm, prev_log_prob_norm );
        vtkm::cont::ArrayHandleConstant<DType> zerosConstHandle(0, numGmms);
		vtkm::cont::Algorithm::Copy( zerosConstHandle, current_log_prob_norm );
        // E Step
		vtkm::cont::Timer EStepTimer;
        vtkm::worklet::DispatcherMapField<EStep> estepDispatcher( EStep{} );
        estepDispatcher.Invoke(trainDataHandle, gmmIdsHandle, resposibilities, log_prob_norms, nonstopFlag, gmmsHandle);
        // Check Stop Condition
		vtkm::cont::Algorithm::ReduceByKey(gmmIdsHandle, log_prob_norms, keyOut_current_log_prob_norm, current_log_prob_norm, vtkm::Add());
        vtkm::worklet::DispatcherMapField<DivideArrayByArray> divideArrayByArrayDispatcher( DivideArrayByArray{} );
        //printf("%f\n", current_log_prob_norm.GetPortalControl().Get(0) );// if I do not print, GPU on OSC is broken (i do not know why)
        divideArrayByArrayDispatcher.Invoke(current_log_prob_norm, numDataInGMM, current_log_prob_norm);
        if( iter != 0 ){ 
            //if it is first iternation, cannot estimate likelihood change
            CheckStop checkstop(stopTol);
            vtkm::worklet::DispatcherMapField<CheckStop> checkStopDispatcher( checkstop );
            checkStopDispatcher.Invoke(current_log_prob_norm, prev_log_prob_norm, nonstopFlag, nonstopFlag);
            vtkm::Int32 sumNonStopFlag = vtkm::cont::Algorithm::Reduce(nonstopFlag, vtkm::Int32(0), vtkm::Add() );
            if( sumNonStopFlag == 0 )break; //stop operation, all GMM converge
        }
        //if( verbose >= 2 )std::cout << "EStepTime: " << EStepTimer.GetElapsedTime() << std::endl;
        // M Step
        ParameterEstimation(numGmms, gmmIdsHandle, resposibilities, keyOut_weights, weights, nonstopFlag, 
                            gmmTrainSampleSnc, meanUpdateIndicator, covUpdateIndicator, newMeans, newCovarianceMats,
                            trainDataHandle, idx2RowHandle, idx2ColHandle, matTriangleSize,
                            verbose, minCovDiag);
    }
  }

    void ParameterEstimation(int numGmms,
                             vtkm::cont::ArrayHandle<vtkm::Int32>& gmmIdsHandle,
                             vtkm::cont::ArrayHandle< vtkm::Vec<DType, GMs> >& resposibilities,
                             vtkm::cont::ArrayHandle<vtkm::Int32>& keyOut_weights,
                             vtkm::cont::ArrayHandle< vtkm::Vec<DType, GMs> > &weights,
                             vtkm::cont::ArrayHandle<vtkm::Int32>& nonstopFlag,
                             vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 2> >& gmmTrainSampleSnc,
                             vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 3> >& meanUpdateIndicator,
                             vtkm::cont::ArrayHandle< vtkm::Vec<vtkm::Int32, 4> >& covUpdateIndicator,
                             vtkm::cont::ArrayHandle< DType >& newMeans,
                             vtkm::cont::ArrayHandle< DType >& newCovarianceMats,
                             vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>>& trainDataHandle,
                             vtkm::cont::ArrayHandle<vtkm::Int32>& idx2RowHandle,
                             vtkm::cont::ArrayHandle<vtkm::Int32>& idx2ColHandle,
                             vtkm::Int32 matTriangleSize,
                             vtkm::Int32 verbose,
                             DType minCovDiag)
    {
        //Update weight
        vtkm::cont::Timer UpdateWeightTimer;
		vtkm::cont::Algorithm::ReduceByKey(gmmIdsHandle, resposibilities, keyOut_weights, weights, vtkm::Add() );
        vtkm::worklet::DispatcherMapField<UpdateWeight> updateWeightDispatcher( UpdateWeight{} );
        updateWeightDispatcher.Invoke(weights, vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numGmms), nonstopFlag, gmmsHandle);
		//if( verbose >=2 )std::cout << "UpdateWeightTime: " << UpdateWeightTimer.GetElapsedTime() << std::endl;

        //Update mean
        vtkm::cont::Timer UpdateMeanTimer;
        vtkm::worklet::DispatcherMapField<ComputeUpdatedMean> computeUpdatedMeanDispatcher( ComputeUpdatedMean{} );
        computeUpdatedMeanDispatcher.Invoke(meanUpdateIndicator, newMeans,
                                            trainDataHandle, gmmTrainSampleSnc, weights,
                                            resposibilities, nonstopFlag);

        vtkm::worklet::DispatcherMapField<UpdateMeanOneShot> updateMeanOneShotDispatcher( UpdateMeanOneShot{} );
        updateMeanOneShotDispatcher.Invoke(vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numGmms),
                                                newMeans,
                                                nonstopFlag,
                                                gmmsHandle );
        //if( verbose >=2 )std::cout << "UpdateMeanTime: " << UpdateMeanTimer.GetElapsedTime() << std::endl;

        //flat cov and update cov
        vtkm::cont::Timer FlatCovMatComputationTimer;
        ComputeUpdatedCovMatrix computeUpdatedCovMatrix(minCovDiag); 
        vtkm::worklet::DispatcherMapField<ComputeUpdatedCovMatrix> computeUpdatedCovMatrixDispatcher( computeUpdatedCovMatrix );
        computeUpdatedCovMatrixDispatcher.Invoke(covUpdateIndicator, newCovarianceMats,
                                                trainDataHandle, gmmTrainSampleSnc, gmmsHandle,
                                                resposibilities, weights, nonstopFlag);
        //if( verbose >=2 )std::cout << "FlatCovMatComputationTime: " << FlatCovMatComputationTimer.GetElapsedTime() << std::endl;
        vtkm::cont::Timer UpdateCovMatrixOneShotTimer;
        UpdateCovMatrixOneShot updateCovMatrixOneShot( matTriangleSize );
        vtkm::worklet::DispatcherMapField<UpdateCovMatrixOneShot> updateCovMatrixOneShotDispatcher( updateCovMatrixOneShot );
        updateCovMatrixOneShotDispatcher.Invoke(vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numGmms),
                                                newCovarianceMats,
                                                nonstopFlag,
                                                gmmsHandle,
                                                idx2RowHandle,
                                                idx2ColHandle );
        //if( verbose >=2 )std::cout << "UpdateCovMatrixOneShotTime: " << UpdateCovMatrixOneShotTimer.GetElapsedTime() << std::endl;
        //Matrix Decompositon (Serial Part)
        vtkm::cont::Timer CovMatDecompositionTimer;
        for( int g=0; g<gmmsHandle.GetNumberOfValues(); g++ ){
            GMM<GMs, VARs, DType> gmm = gmmsHandle.GetPortalControl().Get(g);
            DecomposeCovMatrixToPrecCholMatrix(gmm);
            gmmsHandle.GetPortalControl().Set(g, gmm);
        }
        //if( verbose >=2 )std::cout << "CovMatDecompositionTime (SerialPart): " << CovMatDecompositionTimer.GetElapsedTime() << std::endl << std::endl;

    }

    // Sampling many data point from different GMM
    // parameters: 
    // fromGmmIds: each elements is a gmmId which means the a sample should be drawn from which GMM
    // samples: return samples (length of this array is the same as length of from GmmIds )
    // device: run on which device(GPU, CPU or TBB)
    void SamplingFromMultipleGMMs( std::vector<vtkm::Int32> fromGmmIds,  std::vector<vtkm::Vec<DType, VARs>>& retSamples)
    {
       vtkm::worklet::DispatcherMapField<Sampling> samplingDispatcher( Sampling{} );

        vtkm::cont::ArrayHandle<vtkm::Int32> fromGmmIdsHandle = vtkm::cont::make_ArrayHandle(fromGmmIds);

        std::default_random_engine rng(rand());
        std::uniform_real_distribution<DType> dr(0.0f, 1.0f);
        std::normal_distribution<DType> ndist(0.0f,1.0f);

        std::vector<vtkm::Vec<DType, VARs>> normalRandom;
        std::vector<DType> uniformRandom;

		//generate(begin(normalRandom), end(normalRandom), bind(dr, rng));

        for( int i=0; i<fromGmmIds.size() ; i++ ){
            vtkm::Vec<DType, VARs> p;
            for( int j=0; j<VARs; j++ ){
                p[j] = ndist(rng);
            }
            normalRandom.push_back(p);
            uniformRandom.push_back( dr(rng) );
        }

        vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>> normalRandomHandle = vtkm::cont::make_ArrayHandle(normalRandom);
        vtkm::cont::ArrayHandle<DType> uniformRandomHandle = vtkm::cont::make_ArrayHandle(uniformRandom);

        vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>> samples;
        samplingDispatcher.Invoke(fromGmmIdsHandle, normalRandomHandle, uniformRandomHandle, samples, gmmsHandle);

        retSamples.clear();
        for( int i=0; i<fromGmmIds.size(); i++ ){
           vtkm::Vec<DType, VARs> s = samples.GetPortalControl().Get(i);
           retSamples.push_back( s );
        } 

        //PrintArray(samples);
    }

    void SamplingFromMultipleGMMs( std::vector<vtkm::Int32> fromGmmIds,  vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>>& retSamples)
    {
       vtkm::worklet::DispatcherMapField<Sampling> samplingDispatcher( Sampling{} );

        vtkm::cont::ArrayHandle<vtkm::Int32> fromGmmIdsHandle = vtkm::cont::make_ArrayHandle(fromGmmIds);

        std::default_random_engine rng(rand());
        std::uniform_real_distribution<DType> dr(0.0f, 1.0f);
        std::normal_distribution<DType> ndist(0.0f,1.0f);

        std::vector<vtkm::Vec<DType, VARs>> normalRandom;
        std::vector<DType> uniformRandom;
        for( int i=0; i<fromGmmIds.size() ; i++ ){
            vtkm::Vec<DType, VARs> p;
            for( int j=0; j<VARs; j++ ){
                p[j] = ndist(rng);
            }
            normalRandom.push_back(p);
            uniformRandom.push_back( dr(rng) );
        }

        vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs>> normalRandomHandle = vtkm::cont::make_ArrayHandle(normalRandom);
        vtkm::cont::ArrayHandle<DType> uniformRandomHandle = vtkm::cont::make_ArrayHandle(uniformRandom);

        samplingDispatcher.Invoke(fromGmmIdsHandle, normalRandomHandle, uniformRandomHandle, retSamples, gmmsHandle);
    }

	void testt() {
		std::cout << rand() << std::endl; 
	}
    /////Alocate the gmms array size (use when load gmms from outside e.g. files)
    void AllocateGmmsSize( int numGmms)
    {
        gmmsHandle.Allocate( numGmms );
    }

    //// Set a gmm parameter in gmmsHandle (use when load gmms from outside e.g. files)
    //// This one will set gmm paramters, do eigen decomposition in serial and 
    //// compute U and Eigen matrix for sampling and probability estimate later
    //// Paramters: 
    //// gmmIdx is the location idx in gmmsHandle
    // w (float array): weights of a GMM (length is the same as number of Gaussian components) (from Gau component 0 to GMs-1 )
    // m (float array): means of a GMM (length is the "number of Gaussian components" * "VARs") (from Gau component 0 to GMs-1 )
    // cov(float array): covariance matrixs of a GMM (length is the "number of Gaussian components" * "number of elements of full matrix") (from Gau component 0 to GMs-1 )
    //                   use 3x3 matrix as an example first (assume m20 is the element at row:2 and col:0)
    //                   the order to pass cov should be m00 m01 m02 m10 m11 m12 m20 m21 m22 (row major)
    void SetOneGMM(int gmmIdx, DType *w, DType*m, DType* cov) //set for sampling
    {
      GMM<GMs, VARs, DType> gmm;
      gmm.setGMM(w, m, cov); //1. set w m and cov
      DecomposeCovMatrixToPrecCholMatrix(gmm); //2. call function to decompose cov matrix (pass the gmm)
      //ComputeCovMatInverseAndDet(gmm);  //computer invMat and det is not necessary in EM, just for eays implentation now
      gmmsHandle.GetPortalControl().Set(gmmIdx, gmm); //3. set the gmm to a EM's gmmHandls array
    }

    //SERIAL choleschy decomposition for prob estimation
    void DecomposeCovMatrixToPrecCholMatrix(GMM<GMs, VARs, DType>& gmm)
    {
        for( int c=0; c<GMs; c++ )
        {
            MatrixXd rawCov(VARs, VARs);
            for (int i = 0; i < VARs; i++){
                for (int j = 0; j < VARs; j++){
                    rawCov(i,j) = gmm.covMats[c][i][j];
                }
            }
            
            //choleschy decomposition
            Eigen::LLT<Eigen::MatrixXd>  chol(rawCov);
            if(chol.info()!=Eigen::Success){std::cout << "Choleschy Decomposition Error!!" << std::endl;}
            MatrixXd L = chol.matrixL();
            for (int i = 0; i < VARs; i++){
                for (int j = 0; j < VARs; j++){
                    gmm.lowerMat[c][i][j] = L(i, j);
                }
            }

            MatrixXd I = MatrixXd::Identity(VARs, VARs);
            MatrixXd preciChol = L.triangularView<Lower>().solve(I); //precise choleschy matrix

            // Set preccise choleschy into GMM data structure
            for (int i = 0; i < VARs; i++){
                for (int j = 0; j < VARs; j++){
                    gmm.precCholMat[c][i][j] = preciChol(j,i); //transpose here, so (j,i)
                }
            }

            // Calculate logPDet
            DType logPDet = 0;
            for( int i=0; i<VARs; i++ ){
                logPDet += log( preciChol(i,i) );
            }
            gmm.logPDet[c] = logPDet;
        }       
    }

    void WriteGMMsFile(char *filePath)
    {      
        int gmmCnt = gmmsHandle.GetNumberOfValues();

        std::vector<DType> data;
        data.push_back(gmmCnt);

        for( int i=0; i<gmmCnt; i++ ){
            vtkm::worklet::GMM<GMs, VARs, DType> gmm = gmmsHandle.GetPortalControl().Get(i);

            for( int c=0; c<GMs; c++)data.push_back( gmm.weights[c] );

            for( int c=0; c<GMs; c++){
                for( int v=0; v<VARs; v++ ) data.push_back( gmm.means[c][v] );
            }

            // Only save the lower triangle Cov mat to file
            for( int c=0; c<GMs; c++){
                for( int row = 0; row < VARs; row++ ){
                    for( int col = 0; col <= row; col++ ){
                        data.push_back( gmm.covMats[c][row][col]);
                    }
                }
            }
        }//for(i)
        
        std::ofstream fout(filePath, std::ios::out | std::ios::binary);
        fout.write((char*)&data[0], data.size() * sizeof(DType));
        fout.close();         
    }

    void WriteGMMsFile(char *filePath, int st , int ut) //do not include ut
    {      
        int gmmCnt = ut - st;

        std::vector<DType> data;
        data.push_back(gmmCnt);

        for( int i=st; i<ut; i++ ){
            vtkm::worklet::GMM<GMs, VARs, DType> gmm = gmmsHandle.GetPortalControl().Get(i);

            for( int c=0; c<GMs; c++)data.push_back( gmm.weights[c] );

            for( int c=0; c<GMs; c++){
                for( int v=0; v<VARs; v++ ) data.push_back( gmm.means[c][v] );
            }

            // Only save the lower triangle Cov mat to file
            for( int c=0; c<GMs; c++){
                for( int row = 0; row < VARs; row++ ){
                    for( int col = 0; col <= row; col++ ){
                        data.push_back( gmm.covMats[c][row][col]);
                    }
                }
            }
        }//for(i)
        
        std::ofstream fout(filePath, std::ios::out | std::ios::binary);
        fout.write((char*)&data[0], data.size() * sizeof(DType));
        fout.close();         
    }

    void LoadGMMsFile(char *filePath)
    {
        FILE* fp = fopen( filePath, "rb" ); 
        int gmmCnt;
        DType tmp;
        fread( &tmp, sizeof(DType), 1, fp);
        gmmCnt = tmp;
        
        AllocateGmmsSize( gmmCnt );

        DType w[GMs];
        DType m[GMs*VARs];
        DType cov[GMs*VARs*VARs];
        
        //Compute mapping from 1D linear index to 2D matrix index for lower triagle matrix
        std::vector<int> L2Row, L2Col;
        for( int row = 0; row < VARs; row++ ){
            for( int col = 0; col <= row; col++ ){
                L2Row.push_back( row );
                L2Col.push_back( col );
            }
        }


        int triCovMatSize = ((1+VARs)*VARs)/2;
        for( int i=0; i<gmmCnt; i++ ){
            fread( w, sizeof(DType), GMs, fp );
            fread( m, sizeof(DType), GMs*VARs, fp);

            
            for( int c=0; c<GMs; c++ ){
                for( int j=0; j<triCovMatSize; j++ ){
                    int row = L2Row[j];
                    int col = L2Col[j];
                    fread( &tmp, sizeof(DType), 1, fp);
                    cov[c*VARs*VARs + (row*VARs+col) ] = tmp;
                    if( row != col ) cov[c*VARs*VARs + (col*VARs+row) ] = tmp;
                    
                }
            }
            SetOneGMM(i, w, m, cov);
        }

        fclose( fp );
    }
/*
    void ComputeCovMatInverseAndDet(GMM<GMs, VARs, DType>& gmm)
    {
        //InvMat and det is for conditional distribution
        for( int c=0; c<GMs; c++ )
        {
            MatrixXd rawCov(VARs, VARs); 
            for (int i = 0; i < VARs; i++){
                for (int j = 0; j < VARs; j++){
                    rawCov(i,j) = gmm.covMats[c][i][j];
                }
            }
            MatrixXd invCov = rawCov.inverse();
            //Set inverse matrix 
            for (int i = 0; i < VARs; i++){
                for (int j = 0; j < VARs; j++){
                    gmm.inverseMat[c][i][j] = invCov(i,j);
                }
            }

            gmm.det[c] = rawCov.determinant();
        }
    }*/

};

}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_FieldEntropy_h
