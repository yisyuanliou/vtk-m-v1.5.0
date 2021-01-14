#ifndef vtk_m_worklet_KMeanPP_h
#define vtk_m_worklet_KMeanPP_h

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


namespace vtkm {
namespace worklet {

//simple functor that returns basic statistics
template <int CLUs, int VARs, typename DType>
class KMeanPP
{
public:

    class NormalizedWeight : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        VTKM_CONT
        NormalizedWeight() {}

        template<typename SumWeightWholeArrayInPortalType>
        VTKM_EXEC
        void operator()(const DType &in,
                        const vtkm::Int32 setId,
                        DType &out,
                        SumWeightWholeArrayInPortalType &sumWeightPortal) const
        {
            out = in / sumWeightPortal.Get(setId);
        }
    };


    //Compute distribution based on distance to nearest center
    class ComputeNNCenterDistance : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        vtkm::Int32 NCenters;

        VTKM_CONT
        ComputeNNCenterDistance( vtkm::Int32 _NCenters) :NCenters(_NCenters)
        {}

        template<typename CentersPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs+1> &data, const vtkm::Int32 &setIds, DType &nnSqDist,
                        CentersPortalType &centersPortal ) const
        {
            if( NCenters > 0 ){
                vtkm::Int32 baseCtrIdx = setIds * CLUs;
                DType minSqDistance = std::numeric_limits<DType>::max();
                for( int i=0; i<NCenters; i++ ){
                    DType dist = 0;
                    vtkm::Vec<DType, VARs+1> ctr = centersPortal.Get(baseCtrIdx + i);
                    for( int v=0; v<VARs; v++ ) dist += ( data[v] - ctr[v] ) * ( data[v] - ctr[v] );
                    if( dist < minSqDistance ) minSqDistance = dist;
                }
                nnSqDist = minSqDistance;
            }
            else{//no center is selected
                nnSqDist = 1; 
            }
        }

    };

    class SampleNewCenter : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn,  
                                     WholeArrayIn, WholeArrayIn, WholeArrayIn, WholeArrayOut);
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

        vtkm::Int32 NCenters;

        VTKM_CONT
        SampleNewCenter( vtkm::Int32 _NCenters) :NCenters(_NCenters)
        {}

        template<typename RandomNumberPortalType, typename DistributionPortalType, 
                 typename TrainDataPortalType, typename CentersPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &set, const vtkm::Int32 &setSt, const vtkm::Int32 &setCount,
                        RandomNumberPortalType &randomNumberPortal, DistributionPortalType &distributionPortal,
                        TrainDataPortalType &trainDataPortal, CentersPortalType &centersPortal ) const
        {
            vtkm::Int32 ctrIdx = CLUs*set + NCenters;
            DType r = randomNumberPortal.Get( ctrIdx );
            DType sum = 0;
            int i;
            for( i=0; i<setCount; i++ )
            {
                sum += distributionPortal.Get( setSt + i );
                if( sum > r )break;
            }
            if( i == setCount )i = setCount - 1;
            
            centersPortal.Set( ctrIdx, trainDataPortal.Get( setSt+i ) );
        }
    };

    class AssignLabel : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldOut, FieldOut, 
                                      WholeArrayIn, WholeArrayIn);
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

        VTKM_CONT
        AssignLabel( )
        {}

        template<typename CentersPortalType, typename StopConditionPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs+1> &data, const vtkm::Int32 &setIds, const vtkm::Int32 &oldLabel,
                        vtkm::Int32 &label, vtkm::Int32 &change, CentersPortalType &centersPortal,
                        StopConditionPortalType stopConditionPortal ) const
        {
            vtkm::Int32 stopCondition = stopConditionPortal.Get( setIds );
            if( stopCondition > 0 ){
                vtkm::Int32 ctrSt = CLUs * setIds;
                DType minDist = std::numeric_limits<DType>::max();
                for( int c=0; c<CLUs; c++ )
                {
                    vtkm::Vec<DType, VARs+1> ctr = centersPortal.Get(ctrSt + c );
                    DType dist = 0;
                    for( int v=0; v<VARs; v++ ) dist += ( data[v]-ctr[v] ) * ( data[v]-ctr[v] );
                    //std::cout << c << " "<< sqrt( dist ) <<  "  "<< data << ctr << std::endl;
                    if( dist < minDist ){
                        minDist = dist;
                        label = c;
                    }
                }
                //printf("label %d\n", label);
            
                if( label == oldLabel ) change = 0;
                else change = 1;
            }else{
                change = 0;
                label = oldLabel;
            }
        }

    };

    class ComputeNewMean : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut,
                                     WholeArrayIn, WholeArrayIn, WholeArrayIn, WholeArrayIn,
                                     WholeArrayOut);
        typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);

        VTKM_CONT
        ComputeNewMean( )
        {}

        template<typename TrainDataPortalType, typename LabelPortalType,
                 typename SetStPortalType, typename StopConditionPortalType,
                 typename SetCountPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Int32 &idOfAllCenters, 
                        const vtkm::Vec<DType, VARs> &oldMean, vtkm::Vec<DType, VARs> &newMean,
                        TrainDataPortalType &trainDataPortal, LabelPortalType &labelPortal,
                        SetStPortalType &setStPortal,  StopConditionPortalType stopConditionPortal,
                        SetCountPortalType &setCountPortal ) const
        {
            vtkm::Int32 myLabel = idOfAllCenters % CLUs;
            vtkm::Int32 setId = (idOfAllCenters - myLabel)/CLUs;

            vtkm::Int32 stopCondition = stopConditionPortal.Get( setId );

            if( stopCondition > 0 ){
                vtkm::Int32 setSt = setStPortal.Get(setId);
                vtkm::Int32 setCount = setCountPortal.Get(setId);
            
                vtkm::Float32 inLabelCount = 0;
                vtkm::Vec<DType, VARs> zeroMean(0);
                newMean = zeroMean;
                for( int i=setSt; i<setSt+setCount; i++ )
                {
                    if( myLabel == labelPortal.Get(i) )
                    {
                        inLabelCount ++;
                        newMean = newMean + trainDataPortal.Get( i );
                    }
                }
                newMean = newMean / inLabelCount;
            }
            else{
                newMean = oldMean;
            }
        }
    };

    class ComputeGlobalKey : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2, _3);

        VTKM_CONT
        ComputeGlobalKey( )
        {}

        VTKM_EXEC
        void operator()(const vtkm::Int32 &label, 
                        const vtkm::Int32 &setId, 
                        vtkm::Int32 &globalLabel) const
        {
            globalLabel = label + setId * CLUs;
        }
    };

    class ComputeAssignNewMean : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldIn, WholeArrayOut);
        typedef void ExecutionSignature(_1, _2, _3, _4);

        VTKM_CONT
        ComputeAssignNewMean( )
        {}

        template<typename CenterPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs+1> &sumOfCluster,
                        const vtkm::Int32 &globalKey, 
                        const vtkm::Int32 &numSampleOfCluster,
                        CenterPortalType &centerPortal ) const
        {
            centerPortal.Set( globalKey, sumOfCluster / (float)numSampleOfCluster );
        }
    };

    class CopyTrainArrayWithIndex : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
        typedef void ExecutionSignature(_1, _2, _3);

        VTKM_CONT
        CopyTrainArrayWithIndex( )
        {}

        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs> &trainData,
                        const vtkm::Int32 &index, 
                        vtkm::Vec<DType, VARs+1> &trainDataAppendIndex) const
        {
            for( int i=0; i<VARs; i++ )trainDataAppendIndex[i] = trainData[i];
            trainDataAppendIndex[VARs] = index;
        }
    };

    class CopyBackToOriginalOrder : public vtkm::worklet::WorkletMapField
    {
    public:
        typedef void ControlSignature(FieldIn, FieldIn, WholeArrayOut );
        typedef void ExecutionSignature(_1, _2, _3);

        VTKM_CONT
        CopyBackToOriginalOrder( )
        {}

        template<typename OriginalLabelPortalType>
        VTKM_EXEC
        void operator()(const vtkm::Vec<DType, VARs+1> &trainData,
                        const vtkm::Int32 &label,
                        OriginalLabelPortalType &originalLabalPortal) const
        {
            int originalIndex = trainData[VARs];
            originalLabalPortal.Set(originalIndex, label);
        }
    };

  //set: all train data consists of multiple sets, each set is independet and run indenpendent kmean
  //cluster: data in one set is clustered into N clusters
  //clusterLabel: returned the kmean results
  //Run K-Mean++

  void Run( std::vector<vtkm::Vec<DType, VARs>>& trainData, std::vector<vtkm::Int32>& setIds, vtkm::Int32 numSets, 
            vtkm::Int32 maxIterations, std::vector<vtkm::Int32>& clusterLabel, vtkm::Int32 verbose = 0 )
  {
    //typedef typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter> Algorithm;

    //Self deeply copy and append the original location (the original std vector trainData, should not be changed)
    vtkm::cont::ArrayHandle<vtkm::Vec<DType, VARs+1>> trainDataHandle;
    vtkm::worklet::DispatcherMapField<CopyTrainArrayWithIndex> copyTrainArrayWithIndex( CopyTrainArrayWithIndex{} );
    copyTrainArrayWithIndex.Invoke( vtkm::cont::make_ArrayHandle(trainData),
                                    vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, trainData.size() ),
                                    trainDataHandle );

    vtkm::cont::ArrayHandle<vtkm::Int32> setIdsHandle = vtkm::cont::make_ArrayHandle(setIds);
    vtkm::cont::ArrayHandle< vtkm::Vec<DType, VARs+1> > centers;
    centers.Allocate( CLUs*numSets );
    vtkm::Int32 nTrainData = trainData.size();

    // compute start input data index of each independent set in training Data
    vtkm::cont::ArrayHandle<vtkm::Int32> begSetIdxInData;
    {
        vtkm::cont::ArrayHandleCounting<vtkm::Int32> coutingConstHandle(0, 1, trainDataHandle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<vtkm::Int32> countingHandle;
		vtkm::cont::Algorithm::Copy(coutingConstHandle, countingHandle);
        vtkm::cont::ArrayHandle<vtkm::Int32> keyOut;
		vtkm::cont::Algorithm::ReduceByKey(setIdsHandle, countingHandle, keyOut, begSetIdxInData, vtkm::Minimum());
    }
    // Compute number of train data in each Set
    vtkm::cont::ArrayHandle<vtkm::Int32> numDataInSet;
    {
        vtkm::cont::ArrayHandleConstant<vtkm::Int32> onesConstHandle(1, trainDataHandle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<vtkm::Int32> onesHandle;
		vtkm::cont::Algorithm::Copy(onesConstHandle, onesHandle);
        vtkm::cont::ArrayHandle<vtkm::Int32> keyOut;
		vtkm::cont::Algorithm::ReduceByKey(setIdsHandle, onesHandle, keyOut, numDataInSet, vtkm::Add());
    }

    // random number for cluster center initial selection
    std::vector<DType> uniformRandom;
    for( int i=0; i<CLUs*numSets ; i++ ) uniformRandom.push_back(rand()/(float)RAND_MAX);
    vtkm::cont::ArrayHandle<DType> uniformRandomHandle = vtkm::cont::make_ArrayHandle(uniformRandom);

    //////////////////  K-mean++ initialization ////////////
    vtkm::cont::ArrayHandle< DType > sumWeights;
    vtkm::cont::ArrayHandle< vtkm::Int32 > keyOut;
    vtkm::cont::ArrayHandle< DType > nnSqDist;
    for( int c=0; c<CLUs; c++ ){
        ComputeNNCenterDistance computeNNCenterDistance( c );
        vtkm::worklet::DispatcherMapField<ComputeNNCenterDistance> computeNNCenterDistanceDispatcher( computeNNCenterDistance );
        computeNNCenterDistanceDispatcher.Invoke( trainDataHandle, setIdsHandle, nnSqDist , centers);
		vtkm::cont::Algorithm::ReduceByKey(setIdsHandle, nnSqDist, keyOut, sumWeights, vtkm::Add());
        //normalize weight by sum of weights
        vtkm::worklet::DispatcherMapField<NormalizedWeight> nwDispatcher( NormalizedWeight{} );
        nwDispatcher.Invoke(nnSqDist, setIdsHandle, nnSqDist, sumWeights);
        SampleNewCenter sampleNewCenter( c );
        vtkm::worklet::DispatcherMapField<SampleNewCenter> sampleNewCenterDispatcher( sampleNewCenter );
        sampleNewCenterDispatcher.Invoke( vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, numSets),
                                          begSetIdxInData, numDataInSet,
                                          uniformRandomHandle, nnSqDist, trainDataHandle, centers);
		
    }
    ////////////// K-Mean Algorithm /////////////////////
    std::vector<vtkm::Int32> labels(nTrainData, -1);
    vtkm::cont::ArrayHandle<vtkm::Int32> labelsHandle = vtkm::cont::make_ArrayHandle( labels );
    std::vector<vtkm::Int32> changes(nTrainData, 1);
    vtkm::cont::ArrayHandle<vtkm::Int32> changesHandle = vtkm::cont::make_ArrayHandle( changes) ;
    vtkm::cont::ArrayHandle<vtkm::Int32> setIdsHandleKeyOut;
    vtkm::cont::ArrayHandle<vtkm::Int32> sumOfChangeHandle;
    vtkm::cont::ArrayHandle<vtkm::Int32> globalLabelsHandle;
    vtkm::cont::ArrayHandle< vtkm::Vec<DType, VARs+1> > sumOfSameClusters;
    vtkm::cont::ArrayHandle< vtkm::Int32 > numOfSampleInClusters;
	vtkm::cont::Algorithm::ReduceByKey(setIdsHandle, changesHandle, setIdsHandleKeyOut, sumOfChangeHandle, vtkm::Add());
    for( int iter=0; iter<maxIterations; iter++ )
    {
		//std::cout << iter << std::endl;
        //vtkm::cont::Timer tAssign;
        vtkm::worklet::DispatcherMapField<AssignLabel> assignLabelDispatcher( AssignLabel{} );
        assignLabelDispatcher.Invoke( trainDataHandle, setIdsHandle, labelsHandle, labelsHandle, 
                                        changesHandle, centers, sumOfChangeHandle);
        //std::cout <<  "Assign Time:" << tAssign.GetElapsedTime() << std::endl;
 
		vtkm::cont::Algorithm::ReduceByKey(setIdsHandle, changesHandle, setIdsHandleKeyOut, sumOfChangeHandle, vtkm::Add());
        vtkm::Int32 sumAllChange = vtkm::cont::Algorithm::Reduce(sumOfChangeHandle, vtkm::Int32(0), vtkm::Add() );
        if( sumAllChange == 0 )break;
 
        //THis implementation is fast in GPU, but make TBB a little bit slower
        //vtkm::cont::Timer reduceNewMean;
		
        vtkm::worklet::DispatcherMapField<ComputeGlobalKey> computeGlobalKeyDispatcher( ComputeGlobalKey{} );
        computeGlobalKeyDispatcher.Invoke( labelsHandle, setIdsHandle, globalLabelsHandle);
		vtkm::cont::Algorithm::SortByKey( globalLabelsHandle, trainDataHandle );
		vtkm::cont::Algorithm::ReduceByKey(globalLabelsHandle, trainDataHandle, keyOut, sumOfSameClusters, vtkm::Add());
		vtkm::cont::Algorithm::ReduceByKey(globalLabelsHandle, vtkm::cont::ArrayHandleConstant<vtkm::Int32>(1, globalLabelsHandle.GetNumberOfValues() ),
                                 keyOut, numOfSampleInClusters, vtkm::Add());
        vtkm::worklet::DispatcherMapField<ComputeAssignNewMean> computeAssignNewMeanDispatcher( ComputeAssignNewMean{} );
        computeAssignNewMeanDispatcher.Invoke( sumOfSameClusters, keyOut, numOfSampleInClusters, centers);
		
        // for( int i=0; i<centers.GetNumberOfValues(); i++)std::cout << centers.GetPortalControl().Get(i) << " ";
        // std::cout << " " << std::endl;
        //std::cout <<  "reduce NewMean Time:" << reduceNewMean.GetElapsedTime() << std::endl;

        // This implementation may very slower in GPU, but faster in TBB
        // vtkm::cont::Timer tNewMean;
        // vtkm::worklet::DispatcherMapField<ComputeNewMean> computeNewMeanDispatcher( ComputeNewMean{} );
        // computeNewMeanDispatcher.Invoke( vtkm::cont::ArrayHandleCounting<vtkm::Int32>(0, 1, CLUs*numSets), centers, centers, trainDataHandle, labelsHandle, begSetIdxInData, sumOfChangeHandle, numDataInSet);
        // for( int i=0; i<CLUs; i++)std::cout << centers.GetPortalControl().Get(i) << " ";
        // std::cout << " " << std::endl;
        // std::cout <<  "NewMean Time:" << tNewMean.GetElapsedTime() << std::endl;
    }

    //Run label assignment on trainData with original order
    vtkm::cont::ArrayHandle<vtkm::Int32> originalLabelHandle;
    originalLabelHandle.Allocate( labelsHandle.GetNumberOfValues() );
    vtkm::worklet::DispatcherMapField<CopyBackToOriginalOrder> copyBackToOriginalOrderDispatcher( CopyBackToOriginalOrder{} );
        copyBackToOriginalOrderDispatcher.Invoke( trainDataHandle, labelsHandle, originalLabelHandle );
    // copy labels back to std vector to return
    vtkm::cont::ArrayPortalToIterators<vtkm::cont::ArrayHandle<vtkm::Int32>::PortalControl > 
                                                        iterators( originalLabelHandle.GetPortalControl() );
    clusterLabel.resize(nTrainData);
    std::copy(iterators.GetBegin(), iterators.GetEnd(), clusterLabel.begin ());

  } //End of Run()


};

}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_FieldEntropy_h
