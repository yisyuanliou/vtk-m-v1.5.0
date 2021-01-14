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
//#include <vtkm/worklet/testing/bmp_image.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/worklet/VtkmTable.h>
#include <vtkm/worklet/VtkmSQL.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/cont/DataSetBuilderRectilinear.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/io/reader/VTKDataSetReader.h>

namespace{

  class Sampling : public vtkm::worklet::WorkletMapField
  {
  public:
      typedef void ControlSignature(FieldIn<>, FieldIn<>, FieldIn<>, FieldIn<>, FieldOut<>, WholeArrayIn<>, WholeArrayIn<>, WholeArrayIn<>);
      typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);

      VTKM_CONT
      Sampling() {}

      template<typename BinPortalType, typename FreqPortalType, typename GmmPortalType>
      VTKM_EXEC
      void operator()(const vtkm::Vec<vtkm::Float32,3> xyz,
                      const vtkm::Id &stIdx, const vtkm::Id &len, const vtkm::Float32 &rUniform, 
                      vtkm::Id &sample, BinPortalType &binInPortal, FreqPortalType &freqInPortal,
                      GmmPortalType &gmmInPortal ) const
      {
        vtkm::Float32 postProb[128];
        float postProbSum = 0.0f;
        for( int k=0; k<len; k++ ){
          vtkm::Id f = freqInPortal.Get(stIdx + k);
          postProb[k] = f * gmmInPortal.Get(stIdx + k).getProbability(xyz);
          postProbSum += postProb[k];
        }

        vtkm::Float32 r = postProbSum * rUniform;
        int k;
        vtkm::Float32 sum = 0;
        for( k = 0; k < len; k++ ){
          sum += postProb[k];
          if( sum > r )break;
        }

        if( k == len ) k--;
        sample = binInPortal.Get(stIdx + k);
      }
  };

  class Convert2PointND : public vtkm::worklet::WorkletMapField
  {
  public:
      typedef void ControlSignature(FieldIn<>, FieldIn<>, FieldIn<>, FieldOut<>);
      typedef void ExecutionSignature(_1, _2, _3, _4);

      VTKM_CONT
      Convert2PointND() {}

      VTKM_EXEC
      void operator()(const vtkm::Id &x, const vtkm::Id &y, const vtkm::Id &z,
                      vtkm::Vec<vtkm::Float32,3> &p3d ) const
      {
        p3d[0] = x;
        p3d[1] = y;
        p3d[2] = z;
      }
  };

void TestGMMReconstruction()
{
  // vtkm::Id3 vdims;
  // vtkm::Id blkSize;
  // vtkm::Id nBlocks;
  // vtkm::Id nonZeroBins;
  // vtkm::worklet::GMMTraining<4> em;
  // vtkm::Id* blkStIndex;
  // vtkm::Id* blkLens;
  // vtkm::Id* binidAry;
  // vtkm::Id* freqAry;
  // float* weights;
  // float* means;
  // float* covs;
  // // load file
  // {
  //   FILE *fIn = fopen("gmmHistModel.bin", "rb");
  //   unsigned int ui[6];
  //   fread (ui,sizeof(unsigned int),6,fIn);
  //   vdims[0] = ui[0]; vdims[1] = ui[1]; vdims[2] = ui[2];
  //   blkSize = ui[3];
  //   nBlocks = ui[4];
  //   nonZeroBins = ui[5];

  //   blkStIndex = (vtkm::Id*)malloc(sizeof(vtkm::Id) * nBlocks);
  //   blkLens = (vtkm::Id*)malloc(sizeof(vtkm::Id) * nBlocks);
  //   fread (blkStIndex,sizeof(vtkm::Id), nBlocks,fIn );
  //   fread (blkLens,sizeof(vtkm::Id), nBlocks, fIn );

  //   binidAry = (vtkm::Id*)malloc(sizeof(vtkm::Id) * nonZeroBins);
  //   freqAry = (vtkm::Id*)malloc(sizeof(vtkm::Id) * nonZeroBins);
  //   fread (binidAry, sizeof(vtkm::Id), nonZeroBins, fIn );
  //   fread (freqAry, sizeof(vtkm::Id), nonZeroBins, fIn );

  //   weights = (float*)malloc(sizeof(float) * nonZeroBins * 4); 
  //   means = (float*)malloc(sizeof(float) * nonZeroBins * 4 * 3); 
  //   covs = (float*)malloc(sizeof(float) * nonZeroBins * 4 * 6 ); 
  //   fread (weights, sizeof(float), nonZeroBins * 4, fIn );
  //   fread (means, sizeof(float), nonZeroBins * 4 * 3, fIn );
  //   fread (covs, sizeof(float), nonZeroBins * 4 * 6, fIn );

  //   em.gmmsHandle.Allocate(nonZeroBins);

  //   for( int i=0; i<nonZeroBins; i++ ){
  //     vtkm::worklet::GMM<4> gmm;
  //     gmm.setGMM(weights + i * 4, means + i * 4 * 3, covs + i * 4 * 6);
  //     em.gmmsHandle.GetPortalControl().Set(i, gmm);
  //   }

  //   fclose(fIn);
  // }
  // printf("finish loading\n");

  //load file
  vtkm::Id3 vdims;
  vtkm::Id blkSize;
  vtkm::worklet::GMMTraining<4, 3> em;
  vtkm::cont::DataSet histIndexTable;
  vtkm::cont::DataSet gmmHistTable;
  {
    FILE *fIn = fopen("gmmHistModel_meta.bin", "rb");
    unsigned int ui[4];
    fread (ui,sizeof(unsigned int),4,fIn);
    vdims[0] = ui[0]; vdims[1] = ui[1]; vdims[2] = ui[2];
    blkSize = ui[3];
    fclose(fIn);

    vtkm::worklet::VtkmSQL::VtkmSqlTableReader(histIndexTable, "gmmHistModel_idxTable.vtk" );
    vtkm::worklet::VtkmSQL::VtkmSqlTableReader(gmmHistTable, "gmmHistModel_histGmm.vtk" );
  }
  std::cout << "Finish loading" << std::endl;

  //prepare sampling output
  std::vector<vtkm::Id> rawx, rawy, rawz;
  for( int x=0; x<vdims[0]; x++ ){
    for( int y=0; y<vdims[1]; y++ ){
      for( int z=0; z<vdims[2]; z++ ){
        rawx.push_back(x); 
        rawy.push_back(y); 
        rawz.push_back(z); 
      }
    }
  }
  vtkm::cont::DataSet rawXyzTable;
  vtkm::worklet::VtkmSQL::AddColumn(rawXyzTable, "X", rawx);
  vtkm::worklet::VtkmSQL::AddColumn(rawXyzTable, "Y", rawy);
  vtkm::worklet::VtkmSQL::AddColumn(rawXyzTable, "Z", rawz);

  //Compute block xyz for all xyz we want to reconstruct
  vtkm::cont::DataSet xyzIdx2HistTable;
  {
    vtkm::worklet::VtkmSQL vtkmSql(rawXyzTable, histIndexTable);
    vtkmSql.Select(0, "X", "X" );
    vtkmSql.Select(0, "Y", "Y" );
    vtkmSql.Select(0, "Z", "Z" );
    vtkmSql.Select(0, "X", "QuantizeByMinDelta", "0", "16", "BlkX");
    vtkmSql.Select(0, "Y", "QuantizeByMinDelta", "0", "16", "BlkY");
    vtkmSql.Select(0, "Z", "QuantizeByMinDelta", "0", "16", "BlkZ");
    vtkmSql.Select(1, "BlkX", "BlkX_1");
    vtkmSql.Select(1, "BlkY", "BlkY_1");
    vtkmSql.Select(1, "BlkZ", "BlkZ_1");
    vtkmSql.Select(1, "BlkHistSt", "BlkHistSt" );
    vtkmSql.Select(1, "BlkHistLen", "BlkHistLen");

    vtkmSql.EqualJoin("BlkX", "BlkX_1");
    vtkmSql.EqualJoin("BlkY", "BlkY_1");
    vtkmSql.EqualJoin("BlkZ", "BlkZ_1");

    vtkmSql.SortBy(0, "X" );
    vtkmSql.SortBy(0, "Y" );
    vtkmSql.SortBy(0, "Z" );

    vtkm::cont::Timer<> timer;
    xyzIdx2HistTable = vtkmSql.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
    std::cout<< "Join Xyz and HistGmmIndex Time: " << timer.GetElapsedTime() << std::endl;
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable(xyzIdx2HistTable, 10);
  }

  //Transfer GMM into EM data structure
  {
    vtkm::Id totalBins = vtkm::worklet::VtkmSQL::GetColumnLen(gmmHistTable);
    em.AllocateGmmsSize( totalBins ); /////allocate a GMMs array

    std::vector<std::vector<vtkm::Float32>> gmmArrays(40); //40 is decided by Gaussian compoentns
    for(int i=0; i<gmmArrays.size(); i++ ){
      gmmArrays[i].resize(totalBins);

      std::string fieldName = "Gmm_" + std::to_string(i);
      vtkm::worklet::VtkmSQL::GetColumn( gmmHistTable, fieldName, gmmArrays[i] );
    }

    //convert GMM data to GMM-em internal data structure
    //Example of reading GMM data and set into GMM class
    //**************Several step we need here********************
    //1. Declare vtkm::worklet::GMMTraining<4, 3> em;  (4 is Gaussian components, 3 is variables) em will hold all GMMs
    //2. Set the total number of GMMs which will be loaded by "em.AllocateGmmsSize( numGMMs );"
    //2. load all gmm one by one (maybe from file) 
    //3. for each gmm use em.SetOneGMM() to set a gmm
    //   A. prepare float arrays for weights means and covs  
    //      (GMs->number of Gaussian componenets in a gmm, VARs->number of varaites)
    //          the size of weight is weights[GMs]
    //          the size of means is means[GMs * VARs]
    //          the size of covs is covs[GMs * VARs * VARs]
    //      put the gmm parameters from file to these arrays and call em.SetOneGMM()
    //   B. please refer to SetOneGMM() in GMMTraining.h to get more info abou the parameter
    for( int i=0; i<totalBins; i++ ){ //totalBins here is total number of GMMs
      float weights[4];
      float means[12];
      float covs[36];
      //read weights, means and cov matrix from somewhere and put into the above three array
      for( int j=0; j<4; j++ )weights[j-0] = gmmArrays[j][i];
      for( int j=4; j<16; j++ )means[j-4] = gmmArrays[j][i];
      for( int comp = 0; comp<4; comp++ ){
          int cnt = 0;
        for( int r=0; r<3; r++ ){
          for( int c=r; c<3; c++ ){
            covs[comp*9 + r*3 + c] = gmmArrays[16 + comp*6 + cnt][i];
            covs[comp*9 + c*3 + r] = gmmArrays[16 + comp*6 + cnt][i];
            cnt ++;
          }
        }
      }
      em.SetOneGMM( i, weights, means, covs);
    }
  }

  //Resample from histogram and GMM reconstruction
  {
    vtkm::cont::ArrayHandle<vtkm::Id> binidAry;
    vtkm::worklet::VtkmSQL::GetColumn( gmmHistTable, "Bin", binidAry );
    vtkm::cont::ArrayHandle<vtkm::Id> freqAry;
    vtkm::worklet::VtkmSQL::GetColumn( gmmHistTable, "Freq", freqAry );

    vtkm::cont::ArrayHandle<vtkm::Id> rawX;
    vtkm::worklet::VtkmSQL::GetColumn( xyzIdx2HistTable, "X", rawX );
    vtkm::cont::ArrayHandle<vtkm::Id> rawY;
    vtkm::worklet::VtkmSQL::GetColumn( xyzIdx2HistTable, "Y", rawY );
    vtkm::cont::ArrayHandle<vtkm::Id> rawZ;
    vtkm::worklet::VtkmSQL::GetColumn( xyzIdx2HistTable, "Z", rawZ );
    vtkm::cont::ArrayHandle<vtkm::Id> blkStIndex;
    vtkm::worklet::VtkmSQL::GetColumn( xyzIdx2HistTable, "BlkHistSt", blkStIndex );
    vtkm::cont::ArrayHandle<vtkm::Id> blkLens;
    vtkm::worklet::VtkmSQL::GetColumn( xyzIdx2HistTable, "BlkHistLen", blkLens );

    //Generate Random Number in CPU
    std::default_random_engine rng;
    std::uniform_real_distribution<vtkm::Float32> dr(0.0f, 1.0f);
    std::vector<vtkm::Float32> uniformRandom;
    for( int i=0; i<rawX.GetNumberOfValues() ; i++ ){
        uniformRandom.push_back( dr(rng) );
    }
    vtkm::cont::ArrayHandle<vtkm::Float32> uniformRandomHandle = vtkm::cont::make_ArrayHandle(uniformRandom);

    //Convert to PointND
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>> xyz;
    vtkm::worklet::DispatcherMapField<Convert2PointND, VTKM_DEFAULT_DEVICE_ADAPTER_TAG> convert2PointNDDispatcher( Convert2PointND{} );
    convert2PointNDDispatcher.Invoke(rawX, rawY, rawZ, xyz);

    //Call worklet to sample from histograms
    vtkm::cont::ArrayHandle<vtkm::Id> samples;
    vtkm::worklet::DispatcherMapField<Sampling, VTKM_DEFAULT_DEVICE_ADAPTER_TAG> samplingDispatcher( Sampling{} );
    vtkm::cont::Timer<> timer;
    samplingDispatcher.Invoke(xyz, blkStIndex, blkLens, uniformRandomHandle, samples, binidAry, freqAry, em.gmmsHandle);
    std::cout<< "Data Reconstruction Time: " << timer.GetElapsedTime() << std::endl;

    //write files
    std::vector<vtkm::Float32> fileArray(samples.GetPortalConstControl().GetNumberOfValues());
    std::copy(vtkm::cont::ArrayPortalToIteratorBegin(samples.GetPortalConstControl()),
              vtkm::cont::ArrayPortalToIteratorEnd(samples.GetPortalConstControl()),
              fileArray.begin());
    std::ofstream ofs;
    ofs.open("histGmmReconstruction.bin", std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<char*>(&fileArray[0]), fileArray.size()*sizeof(vtkm::Float32)); 
    ofs.close();
  }

  // //sampling GMM+Hist
  // FILE* fp = fopen( "gmm.bin", "wb" );
  // vtkm::Float32 hist[128];
  // std::vector< vtkm::worklet::GMM<4> > gmms(128);
  // memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // for( int i=0; i< blkStIndex.GetNumberOfValues(); i++ ){
  //   vtkm::Float32 v;

  //   //sampling
  //   //vtkm::Id idx = joinIndex[i];
  //   // vtkm::Id stIdx = blkStIndex[idx];
  //   // vtkm::Id len = blkLens[idx];
  //   vtkm::Id stIdx = blkStIndex.GetPortalConstControl().Get(i);
  //   vtkm::Id len = blkLens.GetPortalConstControl().Get(i);
  //   vtkm::Id histSum = 0;
  //   for( int k=0; k<len; k++ ){
  //     vtkm::Id b = binidAry.GetPortalConstControl().Get(stIdx + k);
  //     vtkm::Id f = freqAry.GetPortalConstControl().Get(stIdx + k);
  //     hist[b] = f;
  //     histSum += f;

  //     gmms[b] = em.gmmsHandle.GetPortalControl().Get(stIdx + k);
  //   }

  //   std::vector< float > postProb(128, 0);
  //   vtkm::Vec<vtkm::Float32,3> PointND;
  //   PointND[0] = rawX.GetPortalConstControl().Get(i); 
  //   PointND[1] = rawY.GetPortalConstControl().Get(i); 
  //   PointND[2] = rawZ.GetPortalConstControl().Get(i);
  //   float postProbSum = 0.0f;
  //   for( int b=0; b<128; b++ ){
  //     if( hist[b] != 0 ){
  //       postProb[b] = hist[b] * gmms[b].getProbability(PointND);
  //       postProbSum +=  postProb[b];
  //     }
  //   }
  //   for( int b=0; b<128; b++ ){              
  //     postProb[b] /= postProbSum;// necessary statement
  //   }

  //   float r = (rand()/(float)RAND_MAX);
  //   float sampleSum = 0;
  //   int b;
  //   for( b=0; b<127; b++ ){
  //     sampleSum += postProb[b];
  //     if( sampleSum > r )break;
  //   }
  //   v = b;

  //   fwrite (&v , sizeof(vtkm::Float32), 1, fp);

  //   memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // }
  // fclose(fp);


  
  // vtkm::cont::DataSet xyz2BlkxyzTable;
  // {
  //   vtkm::cont::DataSet rawXyzTable;
  //   rawXyzTable.AddField(vtkm::cont::Field("X", vtkm::cont::Field::ASSOC_POINTS, rawx));
  //   rawXyzTable.AddField(vtkm::cont::Field("Y", vtkm::cont::Field::ASSOC_POINTS, rawy));
  //   rawXyzTable.AddField(vtkm::cont::Field("Z", vtkm::cont::Field::ASSOC_POINTS, rawz));
  //   std::vector<vtkm::cont::DataSet> db = {rawXyzTable};
  //   vtkm::worklet::VtkmSQL vtkmSql(db);
  //   vtkmSql.Select(0, "X" );
  //   vtkmSql.Select(0, "Y");
  //   vtkmSql.Select(0, "Z");
  //   vtkmSql.Select(0, "X", "QuantizeByMinDelta", "0", "16");
  //   vtkmSql.Select(0, "Y", "QuantizeByMinDelta", "0", "16");
  //   vtkmSql.Select(0, "Z", "QuantizeByMinDelta", "0", "16");
  //   xyz2BlkxyzTable = vtkmSql.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
  // }



  // // vtkm::cont::DataSet xyzIdx2DistriTable;
  // // {
  // //   vtkm::cont::DataSet idx2DistrTable;
  // //   rawXyzTable.AddField(vtkm::cont::Field("StIdx", vtkm::cont::Field::ASSOC_POINTS, blkStIndex));
  // //   rawXyzTable.AddField(vtkm::cont::Field("Len", vtkm::cont::Field::ASSOC_POINTS, blkLens));
  // // }
  // //what join has to do
  // std::vector<vtkm::Id> joinIndex;
  // vtkm::cont::ArrayHandle<vtkm::Id> rawX; xyz2BlkxyzTable.GetField("X").GetData().CopyTo(rawX);
  // vtkm::cont::ArrayHandle<vtkm::Id> rawY; xyz2BlkxyzTable.GetField("Y").GetData().CopyTo(rawY);
  // vtkm::cont::ArrayHandle<vtkm::Id> rawZ; xyz2BlkxyzTable.GetField("Z").GetData().CopyTo(rawZ);
  // vtkm::cont::ArrayHandle<vtkm::Id> rawBlkX; xyz2BlkxyzTable.GetField("QuantizeByMinDelta[0,16](X)").GetData().CopyTo(rawBlkX);
  // vtkm::cont::ArrayHandle<vtkm::Id> rawBlkY; xyz2BlkxyzTable.GetField("QuantizeByMinDelta[0,16](Y)").GetData().CopyTo(rawBlkY);
  // vtkm::cont::ArrayHandle<vtkm::Id> rawBlkZ; xyz2BlkxyzTable.GetField("QuantizeByMinDelta[0,16](Z)").GetData().CopyTo(rawBlkZ);
  // vtkm::Id blkXSize = ceil( vdims[0] / (vtkm::Float32)blkSize );
  // vtkm::Id blkYSize = ceil( vdims[1] / (vtkm::Float32)blkSize );
  // vtkm::Id blkZSize = ceil( vdims[2] / (vtkm::Float32)blkSize );
  // for(int i=0; i<rawBlkX.GetNumberOfValues(); i++ ){
  //     vtkm::Id idx =   
  //       rawBlkX.GetPortalConstControl().Get(i) * (blkYSize * blkZSize) +
  //       rawBlkY.GetPortalConstControl().Get(i) * (blkZSize) +
  //       rawBlkZ.GetPortalConstControl().Get(i);

  //     joinIndex.push_back(idx);
  // }


  // printf("finish prepare\n");

  // // FILE* fp = fopen( "hist.bin", "wb" );
  // // vtkm::Float32 hist[128];
  // // memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // // printf("join index %d\n", joinIndex.size());
  // // for( int i=0; i< joinIndex.size(); i++ ){
  // //   vtkm::Float32 v;

  // //   //sampling
  // //   vtkm::Id idx = joinIndex[i];
  // //   vtkm::Id stIdx = blkStIndex[idx];
  // //   vtkm::Id len = blkLens[idx];
  // //   vtkm::Id histSum = 0;
  // //   for( int k=0; k<len; k++ ){
  // //     vtkm::Id b = binidAry[stIdx + k];
  // //     vtkm::Id f = freqAry[stIdx + k];
  // //     hist[b] = f;
  // //     histSum += f;
  // //   }

  // //   int r = rand() % histSum;
  // //   int sampleSum = 0;
  // //   int b;
  // //   for( b=0; b<127; b++ ){
  // //     sampleSum += hist[b];
  // //     if( sampleSum > r )break;
  // //   }
  // //   v = b;

  // //   fwrite (&v , sizeof(vtkm::Float32), 1, fp);

  // //   memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // // }
  // // fclose(fp);


  // //sampling GMM+Hist
  // FILE* fp = fopen( "gmm.bin", "wb" );
  // vtkm::Float32 hist[128];
  // std::vector< vtkm::worklet::GMM<4> > gmms(128);
  // memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // for( int i=0; i< joinIndex.size(); i++ ){
  //   vtkm::Float32 v;

  //   //sampling
  //   vtkm::Id idx = joinIndex[i];
  //   vtkm::Id stIdx = blkStIndex[idx];
  //   vtkm::Id len = blkLens[idx];
  //   vtkm::Id histSum = 0;
  //   for( int k=0; k<len; k++ ){
  //     vtkm::Id b = binidAry[stIdx + k];
  //     vtkm::Id f = freqAry[stIdx + k];
  //     hist[b] = f;
  //     histSum += f;

  //     gmms[b] = em.gmmsHandle.GetPortalControl().Get(stIdx + k);
  //   }

  //   std::vector< float > postProb(128, 0);
  //   vtkm::Vec<vtkm::Float32,3> PointND;
  //   PointND[0] = rawX.GetPortalConstControl().Get(i); 
  //   PointND[1] = rawY.GetPortalConstControl().Get(i); 
  //   PointND[2] = rawZ.GetPortalConstControl().Get(i);
  //   float postProbSum = 0.0f;
  //   for( int b=0; b<128; b++ ){
  //     if( hist[b] != 0 ){
  //       postProb[b] = hist[b] * gmms[b].getProbability(PointND);
  //       postProbSum +=  postProb[b];
  //     }
  //   }
  //   for( int b=0; b<128; b++ ){              
  //     postProb[b] /= postProbSum;// necessary statement
  //   }

  //   float r = (rand()/(float)RAND_MAX);
  //   float sampleSum = 0;
  //   int b;
  //   for( b=0; b<127; b++ ){
  //     sampleSum += postProb[b];
  //     if( sampleSum > r )break;
  //   }
  //   v = b;

  //   fwrite (&v , sizeof(vtkm::Float32), 1, fp);

  //   memset( hist, 0.0f, sizeof(vtkm::Float32)* 128 );
  // }
  // fclose(fp);

  //std::cout<< "GMM+Hist Reconstruction Time: " << gmmHistReconstructTimer.GetElapsedTime() << std::endl;

} // TestFieldHistogram
}//namespace

int UnitTestGMMReconstruction(int, char* [])
{
  return vtkm::cont::testing::Testing::Run(TestGMMReconstruction);
}
