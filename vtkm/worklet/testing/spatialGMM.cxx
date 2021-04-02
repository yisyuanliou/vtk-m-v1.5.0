//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <iostream>
#include <fstream>
#include <iterator>

#include <vtkm/worklet/FieldHistogram.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/testing/Testing.h>
#include <vtkm/worklet/GMMTraining.h>
#include <vtkm/worklet/KMeanPP.h>
//#include <vtkm/worklet/testing/bmp_image.h>
#include <vtkm/cont/Timer.h>
//#include <vtkm/worklet/VtkmTable.h>
//#include <vtkm/worklet/VtkmSQL.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/RuntimeDeviceTracker.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/cont/DataSetBuilderRectilinear.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/io/reader/VTKDataSetReader.h>

namespace{

typedef vtkm::Float64 Real;
const int VARs = 3;
const int nGauComps = 4;
const int SampleBlockSize = 48;
const int SampleCubeSize = 20;
const int SampleSphereRadius = 100;
const int SampleSphereBand = 5;
const int maxEmIterations = 100;
const int maxKMeanppIterations = 100;
const int vdims[3] = { 500, 500, 100 };
const vtkm::Id numberOfBins = 128;
const vtkm::Float32 fieldMinValue = -4900;
const vtkm::Float32 fieldMaxValue = 2400;
//vtkm::Range range;
vtkm::Float32 delta;
int currentGMMId = -1;
int BinsToGMMId[500000] = {};
int nonZeroBins=0;

class Sampling : public vtkm::worklet::WorkletMapField
{
public:
	typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldIn, FieldOut, WholeArrayIn, WholeArrayIn);
	typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

	VTKM_CONT
		Sampling() {}

	template<typename FreqPortalType, typename GmmPortalType>
	VTKM_EXEC
		void operator()(const vtkm::Vec<vtkm::Float32, 3> xyz,
			const vtkm::Id &stIdx, const vtkm::Id &OrIdx, const vtkm::Float32 &rUniform,
			vtkm::Float32 &sample, FreqPortalType &freqInPortal, GmmPortalType &gmmInPortal) const
	{
		vtkm::Float32 postProb[numberOfBins];
		float postProbSum = 0.0f;
		for (int k = 0; k < numberOfBins; k++) {
			if (BinsToGMMId[stIdx*numberOfBins + k] != -1) {
				//std::cout << stIdx << " d " << BinsToGMMId[stIdx*numberOfBins + k] << std::endl;
				vtkm::Float32 f = freqInPortal.Get(OrIdx*numberOfBins + k);
				//std::cout << OrIdx << " f " << f << std::endl;
				//if(gmmInPortal.Get(BinsToGMMId[stIdx*numberOfBins + k]).getProbability(xyz)!=0) 
					//std::cout<<"gmm: "<<gmmInPortal.Get(BinsToGMMId[stIdx*numberOfBins + k]).getProbability(xyz)<<std::endl;
				vtkm::Float32 Rprob = f * gmmInPortal.Get(BinsToGMMId[stIdx*numberOfBins + k]).getProbability(xyz);
				//std::cout << "Rp " << Rprob << std::endl;
				postProb[k] = Rprob;
				postProbSum += postProb[k];
			}
			else {
				postProb[k] = 0;
			}
		}
		//// Normalize /////
		for (int bin = 0; bin < numberOfBins; ++bin)
		{
			postProb[bin] = postProb[bin] / postProbSum;
		}
		//vtkm::Float32 r = postProbSum * rUniform;
		int k;
		vtkm::Float32 sum = 0;
		for (k = 0; k < numberOfBins; k++) {
			sum += postProb[k];
			if (sum > rUniform)break;
		}
		if (k == numberOfBins) k--;
		//sample = binInPortal.Get(stIdx + k);
		vtkm::Float64 lo = fieldMinValue + (static_cast<vtkm::Float64>(k) * delta);
		vtkm::Float64 hi = lo + delta;
		vtkm::Float64 average = (lo + hi) / 2;
		//std::cout << "s " << k << " " << average << std::endl;
		sample = average;
	}
};

class Convert2PointND : public vtkm::worklet::WorkletMapField
{
public:
	typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldOut);
	typedef void ExecutionSignature(_1, _2, _3, _4);

	VTKM_CONT
		Convert2PointND() {}

	VTKM_EXEC
		void operator()(const vtkm::Id &x, const vtkm::Id &y, const vtkm::Id &z,
			vtkm::Vec<vtkm::Float32, 3> &p3d) const
	{
		p3d[0] = x;
		p3d[1] = y;
		p3d[2] = z;
	}
};

void Index1DTo3D( int idx, int& x, int& y, int& z , const int Size1, const int Size2)
{
    x = idx % Size1;
    idx = (idx - x) / Size1;
    y = idx % Size2;
    z = (idx-y) / Size2;
}

int Index3DTo1D( int x, int y, int z , int size1, int size2)
{
    return z*size1*size2 + y*size1 + x;
}


///////////////////////Important Data Structure ////////////////////////////////
// our GMM trining is designed to train multiple GMM at the same time,
// the number of samples for each GMM does not have to be the same.
// training Data: std::vector< vtkm::Vec<DType, VARs>> >  ex: trainData
// Samples coreespoding GMM ID: std::vector<int>   ex: gmmIds
// trainData, gmmIds, must have the same length
// e.g. trainData[i]'s coresspoding GMM id is gmmIds[i]
// All samples in trainData have same correspoding GMM ID are used to train single GMM
// The samples which have same correspoding GMM ID have to be put together in trainData
// Also, id in gmmIds have to start from 0 and monotonically increase only
void makeDataset(const int *vdims, vtkm::Float32 *data, char* filepath)
{
	FILE* fp = fopen(filepath, "rb");
	if (fp == NULL)
	{
		std::cout << "file open failed!" << std::endl;
	}

	float value;
	int index = 0;
	for (int d0 = 0; d0 < vdims[0]; d0++) {
		for (int d1 = 0; d1 < vdims[1]; d1++) {
			for (int d2 = 0; d2 < vdims[2]; d2++) {
				int tmp = fread(&value, sizeof(float), 1, fp);
				data[index] = value;
				index += 1;
			}
		}
	}
}
vtkm::Int32 findGMMId(vtkm::Id numberOfBins, vtkm::Float32 Minvalue,
	vtkm::Float32 delta, vtkm::Float32 value, int blkCnt)
{
	for (vtkm::Id i = 0; i < numberOfBins; i++)
	{
		vtkm::Float64 lo = Minvalue + (static_cast<vtkm::Float64>(i) * delta);
		vtkm::Float64 hi = lo + delta;
		if (value <= hi && value >= lo)
		{
			return numberOfBins*blkCnt+i;
		}
	}
	return -1;
}
int BinsCntBlocks(vtkm::cont::ArrayHandle<vtkm::Id> bins, vtkm::Id numberOfBins)
{
	int BlkCnt = 0;
	vtkm::cont::ArrayHandle<vtkm::Id>::PortalConstControl binPortal = bins.GetPortalConstControl();
	for (vtkm::Id i = 0; i < numberOfBins; i++)
	{
		if (binPortal.Get(i) > 0)
		{
			BlkCnt += 1;
		}
	}
	return BlkCnt;
}

void TestAyanGMM()
{
	std::string filePrefixPath = "./data/";
	int size = vdims[0] * vdims[1] * vdims[2];
	vtkm::Float32 *gtData = (vtkm::Float32 *)malloc(size * sizeof(vtkm::Float32));
	//vtkm::Float32 *blkCnts = (vtkm::Float32 *)malloc(size * sizeof(vtkm::Float32));
	//vtkm::Float32 *blkOrders = (vtkm::Float32 *)malloc(size * sizeof(vtkm::Float32));
	char pf22file[200] = "pf16.bin";

	// write GMMHistogram data
	std::cout << "write GMMHistogram data" << std::endl;
	std::ofstream GMMHistout(filePrefixPath + "GMMHistModel.bin", std::ios::out | std::ios::binary);
	GMMHistout.write((char*)&vdims[0], 3 * sizeof(int));
	GMMHistout.write((char*)&SampleCubeSize, sizeof(int));
	

	makeDataset(vdims, gtData, pf22file);
	//std::cout << "d" << std::endl;
	int CubeSize = SampleCubeSize * SampleCubeSize*SampleCubeSize;
	int nBlocks = size/CubeSize;
	std::cout<<nBlocks<<std::endl;
	GMMHistout.write((char*)&nBlocks, sizeof(int));
	//int BlockSize = SampleBlockSize * SampleBlockSize * SampleBlockSize;
	//vtkm::Float32 *gtData = (vtkm::Float32 *)malloc(BlockSize * sizeof(vtkm::Float32));
	std::vector<vtkm::Id> TotalBins;

	//BinsToGMMId initialize
	for (int i = 0; i < numberOfBins*(size/CubeSize); ++i)
	{
		BinsToGMMId[i] = -1;
	}

	std::vector<vtkm::Vec<Real, VARs>> gmmTrainData; //******* all training data for all GMMs
	std::vector<vtkm::Int32> gmmIds; //******** sample's coreesponding GMM ID
	//int gmmCnt = numberOfBins*(BlockSize/CubeSize); //*********** how many GMM we want to train
	int gmmCnt = 0; //*********** how many GMM we want to train

	////////////////////  Step 1 ////////////////////////////////
	/////create ground truth data (training data)
	/*for (int i = 0; i < BlockSize; i++) {
		int x, y, z;
		Index1DTo3D(i, x, y, z, SampleBlockSize);
		x += 250;
		y += 200;
		int index = Index3DTo1D(x, y, z, vdims[0]);
		gtData[i] = Data[index];
	}*/
	//Write ground truth to raw binary file for visualization
	//std::ofstream fout(filePrefixPath + "gtData.bin", std::ios::out | std::ios::binary);
	//fout.write((char*)&gtData[0], size * sizeof(vtkm::Float32));
	//fout.close();
	std::cout << "ground truth" << std::endl;

	int blkStIndex[nBlocks];  //not use
    int blkLens[nBlocks];
	int binidAry[26000];      //not fix
	int freqAry[26000];       //not fix
	int nzBinsCnt = 0;
	int Max = 0, Min = 99999;
	for (int a = 0; a < vdims[0] / SampleCubeSize; ++a)
	{
		for (int b = 0; b < vdims[1] / SampleCubeSize; ++b)
		{
			for (int c = 0; c < vdims[2] / SampleCubeSize; ++c)
			{
				/////create a 16*16 block data
				//std::cout << "blockdata" << std::endl;
				blkStIndex[nzBinsCnt] = nzBinsCnt;
				blkLens[nzBinsCnt] = SampleCubeSize;
				nzBinsCnt += 1;
				vtkm::Float32 *blockData = (vtkm::Float32 *)malloc(CubeSize * sizeof(vtkm::Float32));
				for (int d0 = a * SampleCubeSize; d0 < (a + 1)*SampleCubeSize; d0++) {
					for (int d1 = b * SampleCubeSize; d1 < (b + 1)* SampleCubeSize; d1++) {
						for (int d2 = c * SampleCubeSize; d2 < (c + 1)* SampleCubeSize; d2++) {
							int gtIndex = Index3DTo1D(d0, d1, d2, vdims[0], vdims[1]);
							int x = d0 - a * SampleCubeSize;
							int y = d1 - b * SampleCubeSize;
							int z = d2 - c * SampleCubeSize;
							int index = Index3DTo1D(x, y, z, SampleCubeSize, SampleCubeSize);
							blockData[index] = gtData[gtIndex];
							if (gtData[gtIndex] > Max)
							{
								Max = gtData[gtIndex];
							}
							if (gtData[gtIndex] < Min)
							{
								Min = gtData[gtIndex];
							}
						}
					}
				}
				//std::cout << "finish" << std::endl;
				////////  Histogram  //////////
				// Create the output bin array
				vtkm::cont::ArrayHandle<vtkm::Id> bins;
				bins.Allocate(numberOfBins);
				// Get point data
				vtkm::cont::ArrayHandle<vtkm::Float32> p_data = vtkm::cont::make_ArrayHandle(blockData, CubeSize);

				vtkm::worklet::FieldHistogram histogram;
				// Run data

				histogram.Run(p_data, numberOfBins, fieldMinValue, fieldMaxValue, delta, bins);
				//histogram.Run(p_data, numberOfBins, range, delta, bins);
				gmmCnt += BinsCntBlocks(bins, numberOfBins);
				TotalBins.insert(TotalBins.end(), vtkm::cont::ArrayPortalToIteratorBegin(bins.GetPortalConstControl()),
					vtkm::cont::ArrayPortalToIteratorEnd(bins.GetPortalConstControl()));
				//std::cout << "histogram" << std::endl;
				////////////count histogram/////////
				//int tmp=0;
				for(int bin = 0 ; bin < numberOfBins ; ++bin) {
					if((int)bins.GetPortalConstControl().Get(bin)!=0) {
						binidAry[nonZeroBins] = bin;
						freqAry[nonZeroBins] = (int)bins.GetPortalConstControl().Get(bin);
						//std::cout<<bin<<" ";
						//tmp += 1;
						nonZeroBins += 1;
					}
				}//std::cout<<tmp<<std::endl;
				////////////////////// Step 2: data to gmm training format ///////////////////////////////////////
				///// from blockData to GMM traiing format
				for (int d0 = a * SampleCubeSize; d0 < (a + 1)*SampleCubeSize; d0++) {
					for (int d1 = b * SampleCubeSize; d1 < (b + 1)* SampleCubeSize; d1++) {
						for (int d2 = c * SampleCubeSize; d2 < (c + 1)* SampleCubeSize; d2++) {
							vtkm::Vec<Real, VARs> sample;
							sample[0] = d0; sample[1] = d1; sample[2] = d2;
							gmmTrainData.push_back(sample);
							int x = d0 - a * SampleCubeSize;
							int y = d1 - b * SampleCubeSize;
							int z = d2 - c * SampleCubeSize;
							int bkIndex = Index3DTo1D(x, y, z, SampleCubeSize, SampleCubeSize);
							int blkCnt = Index3DTo1D(a, b, c, vdims[0] / SampleCubeSize, vdims[1] / SampleCubeSize);
							
							//int gtIndex = Index3DTo1D(d0, d1, d2, vdims[0], vdims[1]);
							//blkCnts[gtIndex] = blkCnt;
							int binsId = findGMMId(numberOfBins, fieldMinValue, delta, blockData[bkIndex], blkCnt);


							if (BinsToGMMId[binsId] == -1)
							{
								currentGMMId += 1;
								BinsToGMMId[binsId] = currentGMMId;
								//std::cout<<" "<<currentGMMId<<" ";
								gmmIds.push_back(currentGMMId);
							}
							else
							{
								gmmIds.push_back(BinsToGMMId[binsId]);
							}
						}
					}
				}
				//std::cout<<std::endl;
				free(blockData);
			}
		}
	}
	std::cout << currentGMMId << " "<<gmmCnt<<" ";
	std::cout << "Max: " << Max << " Min: " << Min << std::endl;

	GMMHistout.write((char*)&nonZeroBins, sizeof(int));
	GMMHistout.write((char*)&blkStIndex[0], nBlocks * sizeof(int));
	GMMHistout.write((char*)&blkLens[0], nBlocks * sizeof(int));
	GMMHistout.write((char*)&binidAry[0], nonZeroBins * sizeof(int));
	GMMHistout.write((char*)&freqAry[0], nonZeroBins * sizeof(int));
	GMMHistout.write((char*)&BinsToGMMId[0], numberOfBins*nBlocks * sizeof(int));
	GMMHistout.close();

	// for(int i=0;i<10;++i)
	// {
	// 	std::cout<<"bin: "<<binidAry[i]<<" "<<freqAry[i]<<std::endl;
	// }
	// for(int i=0;i<100;++i)
	// {
	// 	std::cout<<"bin2array: "<<BinsToGMMId[i]<<std::endl;
	// }

	
	// Normalize Histogram data
	vtkm::cont::ArrayHandle<vtkm::Float32> Hbins;
	Hbins.Allocate(TotalBins.size());

	for (int i = 0; i < TotalBins.size(); ++i)
	{
		vtkm::Float32 value = (vtkm::Float32)TotalBins[i] / (vtkm::Float32)CubeSize;
		Hbins.GetPortalControl().Set(i, value);
	}
	//std::cout << "cnt: " << gmmCnt << std::endl;

	vtkm::cont::ArrayHandle < vtkm::Int32 > keys = vtkm::cont::make_ArrayHandle(gmmIds);
	vtkm::cont::ArrayHandle < vtkm::Vec<Real, VARs> > values = vtkm::cont::make_ArrayHandle(gmmTrainData);

	vtkm::cont::Algorithm::SortByKey(keys, values);

	std::copy(vtkm::cont::ArrayPortalToIteratorBegin(keys.GetPortalConstControl()),
		vtkm::cont::ArrayPortalToIteratorEnd(keys.GetPortalConstControl()),
		gmmIds.begin());
	std::copy(vtkm::cont::ArrayPortalToIteratorBegin(values.GetPortalConstControl()),
		vtkm::cont::ArrayPortalToIteratorEnd(values.GetPortalConstControl()),
		gmmTrainData.begin());
	std::cout << "size: "<< gmmIds.size() << std::endl;

	//std::cout << gmmTrainData.size() << " " << gmmIds.size() << std::endl;
	/////////////////////// Step 3: train  /////////////////////////////////////////////////
	// Run Kmean++ first to get better initialization for GMM training
	vtkm::cont::Timer kmeanTimer;
	kmeanTimer.Start();
	//std::cout << "ks" << std::endl;

	std::vector< vtkm::Int32 > clusterLabel;

	vtkm::worklet::KMeanPP<nGauComps, VARs, Real> kmeanpp;
	kmeanpp.Run(gmmTrainData, gmmIds, gmmCnt, maxKMeanppIterations, clusterLabel);
	std::cout << "Kmean++" << std::endl;
	kmeanTimer.Stop();

	std::cout<< "KMean++ Time: " << kmeanTimer.GetElapsedTime() << std::endl << std::endl;

	//// GMM Training EM
	srand(time(NULL));
	vtkm::cont::Timer gmmTimer;
	gmmTimer.Start();
	vtkm::worklet::GMMTraining<nGauComps, VARs, Real> em;
	em.Run(gmmTrainData, gmmIds, gmmCnt, maxEmIterations, clusterLabel, 1, 2, 0.001, 0.000001);
	std::cout << "EM finish" << std::endl;
	gmmTimer.Stop();

	std::cout << "GMM Time: " << gmmTimer.GetElapsedTime() << std::endl << std::endl;

	//6th para: 0: random init (clusterLabel useless), 1: use kmean++ init
	//7th para: screen output: 0: no output, 1:only iteration output, 2: output details
	//8th para: em stop threshold
	//9th para: minimal diagnal value in covariance matrix value (em must add this value to diagonal in cov mat)

	
	//////////////////////// Step 4 : write to file //////////////////////////////////
	char gmmFilePath[1000];
	strcpy(gmmFilePath, filePrefixPath.c_str());
	strcat(gmmFilePath, "gmm.bin");
	em.WriteGMMsFile(gmmFilePath); ////  Save GMM to a file

	//char gmmFilePath[1000]="gmm.bin";
	//////////////////////// Step 5 : load gmms from file /////////////////////////////////
	//// Load GMM from a file, programmer has to know the nGauComps, VARs, Real used to train the GMM
	vtkm::worklet::GMMTraining<nGauComps, VARs, Real> rsp;
	rsp.LoadGMMsFile(gmmFilePath);
	std::cout << "check " << rsp.gmmsHandle.GetNumberOfValues()<< std::endl;

	//////////////////////// Step 6 : reconstruction /////////////////////////////////
	//prepare sampling output
	std::vector<vtkm::Id> rawx, rawy, rawz;
	std::vector<vtkm::Id> blockCnts, blockOrders;
	for (int i = 0; i < size; i++) {
		int x, y, z;
		Index1DTo3D(i, x, y, z, vdims[0], vdims[1]);

		int blockCnt = Index3DTo1D(x / SampleCubeSize, y / SampleCubeSize, z / SampleCubeSize, vdims[0]/SampleCubeSize, vdims[1] / SampleCubeSize);
		blockCnts.push_back(blockCnt);

		int blockOrder = Index3DTo1D(z / SampleCubeSize, y / SampleCubeSize, x / SampleCubeSize, vdims[2] / SampleCubeSize, vdims[1] / SampleCubeSize);
		blockOrders.push_back(blockOrder);

		rawx.push_back(x);
		rawy.push_back(y);
		rawz.push_back(z);
	}
	vtkm::cont::ArrayHandle<vtkm::Id> rawX = vtkm::cont::make_ArrayHandle(rawx);
	vtkm::cont::ArrayHandle<vtkm::Id> rawY = vtkm::cont::make_ArrayHandle(rawy);
	vtkm::cont::ArrayHandle<vtkm::Id> rawZ = vtkm::cont::make_ArrayHandle(rawz);

	vtkm::cont::ArrayHandle<vtkm::Id> blkCntIndex = vtkm::cont::make_ArrayHandle(blockCnts);
	vtkm::cont::ArrayHandle<vtkm::Id> blkOrIndex = vtkm::cont::make_ArrayHandle(blockOrders);

	//Convert to PointND
	vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>> xyz;
	vtkm::worklet::DispatcherMapField<Convert2PointND> convert2PointNDDispatcher(Convert2PointND{});
	convert2PointNDDispatcher.Invoke(rawX, rawY, rawZ, xyz);

	//Generate Random Number in CPU
	std::default_random_engine rng;
	std::uniform_real_distribution<vtkm::Float32> dr(0.0f, 1.0f);
	std::vector<vtkm::Float32> uniformRandom;
	for (int i = 0; i < rawX.GetNumberOfValues(); i++) {
		uniformRandom.push_back(dr(rng));
	}
	vtkm::cont::ArrayHandle<vtkm::Float32> uniformRandomHandle = vtkm::cont::make_ArrayHandle(uniformRandom);
	//Call worklet to sample from histograms
	vtkm::cont::ArrayHandle<vtkm::Float32> samples;
	samples.Allocate(size);

	vtkm::cont::Timer resampleTimer;
	vtkm::worklet::DispatcherMapField<Sampling> samplingDispatcher(Sampling{});
	resampleTimer.Start();
	samplingDispatcher.Invoke(xyz, blkCntIndex, blkOrIndex, uniformRandomHandle, samples, Hbins, rsp.gmmsHandle);
	std::cout << "Data Reconstruction Time: " << resampleTimer.GetElapsedTime() << std::endl;

	//std::cout << "check" << std::endl;
	//std::cout << ValueHandle.GetPortalConstControl().GetNumberOfValues() << std::endl;
	std::vector<vtkm::Float32> reSampleData(samples.GetPortalConstControl().GetNumberOfValues());
	std::copy(vtkm::cont::ArrayPortalToIteratorBegin(samples.GetPortalConstControl()),
		vtkm::cont::ArrayPortalToIteratorEnd(samples.GetPortalConstControl()),
		reSampleData.begin());
	
	GMMHistout.close();
	std::ofstream ofout(filePrefixPath + "reSampleData.bin", std::ios::out | std::ios::binary);
	ofout.write((char*)&reSampleData[0], reSampleData.size() * sizeof(float));
	ofout.close();
	free(gtData);

} // TestMain Function
}//namespace

int spatialGMM(int argc, char* argv[])
{
	vtkm::cont::GetRuntimeDeviceTracker().ForceDevice(vtkm::cont::DeviceAdapterTagTBB{});
	// vtkm::cont::ScopedRuntimeDeviceTracker(vtkm::cont::DeviceAdapterTagSerial{});
    return vtkm::cont::testing::Testing::Run(TestAyanGMM, argc, argv);
}
