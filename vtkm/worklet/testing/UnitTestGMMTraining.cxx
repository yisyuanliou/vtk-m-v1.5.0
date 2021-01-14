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

vtkm::cont::DataSet Make3DDataSet( vtkm::Id3& vdims, char* filepath )
{
  vtkm::cont::DataSet dataSet;

  std::cout << vdims[0] << vdims[1] << vdims[2] << std::endl;
  vtkm::Id size = vdims[0] * vdims[1] * vdims[2];

  std::vector<vtkm::Id> x;
  std::vector<vtkm::Id> y;
  std::vector<vtkm::Id> z;
  std::vector<vtkm::Float32> v;

  std::cout << filepath << std::endl;
  FILE* fp = fopen(filepath, "rb");

  float value;
  for( int d0=0; d0<vdims[0]; d0++){
    for( int d1=0; d1<vdims[1]; d1++){
      for( int d2=0; d2<vdims[2]; d2++){
        int tmp = fread(&value, sizeof(float), 1, fp);
        x.push_back(d0);
        y.push_back(d1);
        z.push_back(d2);
        v.push_back(value);
      }
    }
  }

  vtkm::worklet::VtkmSQL::AddColumn(dataSet, "X", x);
  vtkm::worklet::VtkmSQL::AddColumn(dataSet, "Y", y);
  vtkm::worklet::VtkmSQL::AddColumn(dataSet, "Z", z);
  vtkm::worklet::VtkmSQL::AddColumn(dataSet, "V", v);

  return dataSet;
}

void TestGMMTraining()
{
  // Isabel
  vtkm::Id3 vdims(100, 500, 500);
  char filepath[200] = "../../DataSets/isabel/pf20.bin";
  //char filepath[200] = "C:\\GravityLabDataSet\\IsabelTimeVarying\\Pf\\LeNoNan\\pf20.bin";
  // Combustion
  // vtkm::Id3 vdims(120, 720, 480);
  // char filepath[200] = "/home/caseywng777/Desktop/jet_mixfrac_0040.bin";
  vtkm::Id blkSize = 16;
  vtkm::cont::DataSet isabelRawDs = Make3DDataSet( vdims,  filepath );

  vtkm::Id gmmCnt;
  std::vector<vtkm::Int32> gmmIds;
  std::vector< vtkm::Vec<vtkm::Float32, 3> > trainData;
  vtkm::cont::DataSet histogramTable;
{
  // Compute Histogram
  
  {
    vtkm::worklet::VtkmSQL vtkmSql(isabelRawDs);
    vtkmSql.Select(0, "X", "QuantizeByMinDelta", "0", "16", "BlkX");
    vtkmSql.Select(0, "Y", "QuantizeByMinDelta", "0", "16", "BlkY");
    vtkmSql.Select(0, "Z", "QuantizeByMinDelta", "0", "16", "BlkZ");
    vtkmSql.Select(0, "V", "QuantizeByBin", "128", "Bin");
    vtkmSql.Select(0, "RowID", "COUNT", "Freq" );

    vtkmSql.GroupBy(0, "BlkX");
    vtkmSql.GroupBy(0, "BlkY");
    vtkmSql.GroupBy(0, "BlkZ");
    vtkmSql.GroupBy(0, "Bin");

    vtkmSql.SortBy(0, "BlkX");
    vtkmSql.SortBy(0, "BlkY");
    vtkmSql.SortBy(0, "BlkZ");
    vtkmSql.SortBy(0, "Bin");

    vtkm::cont::Timer<> timer;
    histogramTable = vtkmSql.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
    std::cout<< "Histogram Computation Time: " << timer.GetElapsedTime() << std::endl;
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable(histogramTable, 10);
  }

  // Collect and prepare training data for GMM trainer
  vtkm::cont::DataSet gmmTrainDataTable;
  {
    vtkm::worklet::VtkmSQL vtkmSql(isabelRawDs, histogramTable);
    vtkmSql.Select(0, "X", "X" );
    vtkmSql.Select(0, "Y", "Y" );
    vtkmSql.Select(0, "Z", "Z" );
    vtkmSql.Select(0, "X", "QuantizeByMinDelta", "0", "16", "BlkX");
    vtkmSql.Select(0, "Y", "QuantizeByMinDelta", "0", "16", "BlkY");
    vtkmSql.Select(0, "Z", "QuantizeByMinDelta", "0", "16", "BlkZ");
    vtkmSql.Select(0, "V", "QuantizeByBin", "128", "Bin");
    vtkmSql.Select(1, "BlkX", "BlkX_1" );
    vtkmSql.Select(1, "BlkY", "BlkY_1");
    vtkmSql.Select(1, "BlkZ", "BlkZ_1");
    vtkmSql.Select(1, "Bin", "Bin_1");
    vtkmSql.Select(1, "RowID", "GmmID");

    vtkmSql.EqualJoin("BlkX", "BlkX_1");
    vtkmSql.EqualJoin("BlkY", "BlkY_1");
    vtkmSql.EqualJoin("BlkZ", "BlkZ_1");
    vtkmSql.EqualJoin("Bin", "Bin_1");

    vtkmSql.SortBy(1, "GmmID");

    vtkm::cont::Timer<> timer;
    gmmTrainDataTable = vtkmSql.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
    std::cout<< "Gmm Training Data Collection Time: " << timer.GetElapsedTime() << std::endl;
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable(gmmTrainDataTable, 10);
  }

  // Convert data from VtkmSQL to GMM-EM format, then train
  
  //vtkm::Id gmmCnt;
  {
    vtkm::cont::ArrayHandle<vtkm::Id> xAry; 
    vtkm::worklet::VtkmSQL::GetColumn(gmmTrainDataTable, "X", xAry);
    vtkm::cont::ArrayHandle<vtkm::Id> yAry; 
    vtkm::worklet::VtkmSQL::GetColumn(gmmTrainDataTable, "Y", yAry);
    vtkm::cont::ArrayHandle<vtkm::Id> zAry; 
    vtkm::worklet::VtkmSQL::GetColumn(gmmTrainDataTable, "Z", zAry);
    vtkm::cont::ArrayHandle<vtkm::Id> idAry; 
    vtkm::worklet::VtkmSQL::GetColumn(gmmTrainDataTable, "GmmID", idAry);
    //no parallize slow here
    for( int i=0; i<xAry.GetNumberOfValues(); i++ ){
      vtkm::Vec<vtkm::Float32,3> coordi3D;
      coordi3D[0] = xAry.GetPortalConstControl().Get(i);
      coordi3D[1] = yAry.GetPortalConstControl().Get(i);
      coordi3D[2] = zAry.GetPortalConstControl().Get(i);
      trainData.push_back(coordi3D);
      gmmIds.push_back( idAry.GetPortalConstControl().Get(i) );
    }
    
    gmmCnt = vtkm::worklet::VtkmSQL::GetColumnLen(histogramTable);
  }

}

  /////GMM training starts
  //vtkm::worklet::GMMTraining<4, 3> em;
  // 4 indicates each GMM have 4 Gaussian components, 
  // 3 (GMs) indicates that training three varaibles GMM
  vtkm::worklet::GMMTraining<4, 3> em; 
  vtkm::cont::Timer<> gmmTimer;
  //parameters:
  //trainData(std::vector< vtkm::worklet::vtkm::Vec<vtkm::Float32, GMs> >) : training samples
  //                              GMs is the number of variate of a training sample (this example is 3)
  //                              std::vector is whole training samples which will be trainned for all GMMs
  //                              vtkm::worklet::PointND is a N-variable training sample
  //gmmIds(std::vector<vtkm::Int32>) : a sample is correspoding to which GMM
  //                              each elements in this vector is a id(integer)
  //                              trainData and gmmIds must have the same length
  //                              each value in this vector indicates the correspoding sample in trainData's gmmId
  //                              (GMM ID should start from 0 and increase)
  //                              The samples in trainData which will generate a GMM should have the same gmmId
  //gmmCnt(int): total number of GMMs
  //the four parameter(int): maximal number of EM iteration you allow (I usually use 10~30)
  //VTKM_DEFAULT_DEVICE_ADAPTER_TAG(): run on which device (GPU CPU or TBB)
  //*****NOTE: the gmmId in "gmmIds" must be monotonically increase from 0 without id jumpping********
  //*********the id in gmmIds should looks like this 0 0 0 0 1 2 2 3 3 3 3 3 4 4...............
  //*********therefore, the samples in "trainData" to train one GMM must be put in consective array locations
  em.Run(trainData, gmmIds, gmmCnt, 10, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
   
  std::cout<< "GMM EM Time: " << gmmTimer.GetElapsedTime() << std::endl;
  ///// GMM training finish

  //Compute index for the sparse representation histograms
  vtkm::cont::DataSet histIndexTable;
  {
    vtkm::worklet::VtkmSQL vtkmSql(histogramTable);
    vtkmSql.Select(0, "BlkX", "BlkX" );
    vtkmSql.Select(0, "BlkY", "BlkY" );
    vtkmSql.Select(0, "BlkZ", "BlkZ" );
    vtkmSql.Select(0, "RowID", "MIN", "BlkHistSt" );
    vtkmSql.Select(0, "RowID", "COUNT", "BlkHistLen" );

    vtkmSql.GroupBy(0, "BlkX" );
    vtkmSql.GroupBy(0, "BlkY" );
    vtkmSql.GroupBy(0, "BlkZ" );

    vtkmSql.SortBy(0, "BlkX" );
    vtkmSql.SortBy(0, "BlkY" );
    vtkmSql.SortBy(0, "BlkZ" );

    vtkm::cont::Timer<> timer;
    histIndexTable = vtkmSql.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
    std::cout<< "Index Table Computation Time: " << timer.GetElapsedTime() << std::endl;
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable(histIndexTable, 10);
  }

  //New format: output data structure to file
  {
    std::string mainName = "gmmHistModel_";
    std::string metaName = "meta.bin";
    std::string idxTableName = "idxTable.vtk";
    std::string histGmmName = "histGmm.vtk";
    
    //meta data
    FILE* fOut = fopen( (mainName + metaName).c_str(), "wb");
    unsigned int ui[4];
    ui[0] = vdims[0]; ui[1] = vdims[1]; ui[2] = vdims[2]; //raw xyz
    ui[3] = blkSize; //block size;
    fwrite (ui , sizeof(unsigned int), 4, fOut);
    fclose(fOut);
    std::cout << "Write :" << mainName + metaName << std::endl;

    //index table: blk xyz -> st and len of hist and gmm
    vtkm::worklet::VtkmSQL::VtkmSqlTableWriter(histIndexTable, mainName + idxTableName );
    std::cout << "Write :" << mainName + idxTableName << std::endl;

    //write gmm and hist table
    vtkm::cont::DataSet histGmmTable;

    vtkm::cont::ArrayHandle<vtkm::Id> bins;
    vtkm::worklet::VtkmSQL::GetColumn(histogramTable, "Bin", bins);
    vtkm::worklet::VtkmSQL::AddColumn(histGmmTable, "Bin", bins );

    vtkm::cont::ArrayHandle<vtkm::Id> freqs;
    vtkm::worklet::VtkmSQL::GetColumn(histogramTable, "Freq", freqs);
    vtkm::worklet::VtkmSQL::AddColumn(histGmmTable,  "Freq", freqs );

    ///// Example to read GMMs from EM training/////
    //// THe example here is read GMMs with 3 variables and 4 gaussian components
    //// Use vtkm::worklet::GMM<4, 3> gmm = em.gmmsHandle.GetPortalControl().Get(i); to get the i-th gmm (4 Gaussians 3 variable case)
    //// then, gmm.weights[c] is the weight of the c-th Gaussian component in gmm
    //// gmm.means[c][v] is the v-th variable of the mean of c-th Gaussian in gmm
    //// gmm.covMats[c][j][k] is the value at row-j and column-k of the c-th component's covariance matrix in gmm
    std::vector<std::vector<vtkm::Float32>> gmmArrays(40); //4 gaussian per GMM, each Gaussian has 10 float(w, mmm, cccccc)
    /////gmm Array is just my personal file format to write GMMs to a file
    for( int i=0; i<gmmCnt; i++ ){ //go through all GMMs
      vtkm::worklet::GMM<4, 3> gmm = em.gmmsHandle.GetPortalControl().Get(i); //get a gmm
      for( int c=0; c<4; c++){ //read all weights of the gmm
        gmmArrays[0 + c].push_back( gmm.weights[c] );
      }

      for( int c=0; c<4; c++){ //read all means of the gmm
        gmmArrays[4 + c*3 + 0].push_back( gmm.means[c][0] );
        gmmArrays[4 + c*3 + 1].push_back( gmm.means[c][1] );
        gmmArrays[4 + c*3 + 2].push_back( gmm.means[c][2] );
      }

      for( int c=0; c<4; c++){ //read all cov matrixs of the gmm
        gmmArrays[16 + c*6 + 0].push_back( gmm.covMats[c][0][0] );
        gmmArrays[16 + c*6 + 1].push_back( gmm.covMats[c][0][1] );
        gmmArrays[16 + c*6 + 2].push_back( gmm.covMats[c][0][2] );
        gmmArrays[16 + c*6 + 3].push_back( gmm.covMats[c][1][1] );
        gmmArrays[16 + c*6 + 4].push_back( gmm.covMats[c][1][2] );
        gmmArrays[16 + c*6 + 5].push_back( gmm.covMats[c][2][2] );
      }
    }
    ///// End of reading example

    for( int i=0; i<gmmArrays.size(); i++ ){
      std::string fieldName = "Gmm_" + std::to_string(i);
      std::cout << fieldName << std::endl;
      vtkm::worklet::VtkmSQL::AddColumn(histGmmTable, fieldName, gmmArrays[i] );
    }

    vtkm::worklet::VtkmSQL::VtkmSqlTableWriter(histGmmTable, mainName + histGmmName );

    std::cout << "NGMM: " << histGmmTable.GetField(0).GetData().GetNumberOfValues() << std::endl;
    std::cout << "Write :" << mainName + histGmmName << std::endl;
  }





  // // output our data structure to file
  // {
  //   //vtkm::Id numBlocks = histIndexTable.GetField("QuantizeByMinDelta[0,16](X)").GetData().GetNumberOfValues();
  //   vtkm::cont::ArrayHandle<vtkm::Id> blkStIndexAryHandle; histIndexTable.GetField("MIN(ID)").GetData().CopyTo(blkStIndexAryHandle);
  //   std::vector<vtkm::Id> blkStIndex(blkStIndexAryHandle.GetPortalConstControl().GetNumberOfValues());
  //   std::copy(vtkm::cont::ArrayPortalToIteratorBegin(blkStIndexAryHandle.GetPortalConstControl()),
  //             vtkm::cont::ArrayPortalToIteratorEnd(blkStIndexAryHandle.GetPortalConstControl()),
  //             blkStIndex.begin());
  //   vtkm::cont::ArrayHandle<vtkm::Id> blkLenAryHandle; histIndexTable.GetField("COUNT(ID)").GetData().CopyTo(blkLenAryHandle);
  //   std::vector<vtkm::Id> blkLensVec(blkLenAryHandle.GetPortalConstControl().GetNumberOfValues());
  //   std::copy(vtkm::cont::ArrayPortalToIteratorBegin(blkLenAryHandle.GetPortalConstControl()),
  //             vtkm::cont::ArrayPortalToIteratorEnd(blkLenAryHandle.GetPortalConstControl()),
  //             blkLensVec.begin());
  //   vtkm::cont::ArrayHandle<vtkm::Id> binidAry; histogramTable.GetField("QuantizeByBin[128](V)").GetData().CopyTo(binidAry);
  //   vtkm::cont::ArrayHandle<vtkm::Id> freqAry; histogramTable.GetField("COUNT(ID)").GetData().CopyTo(freqAry);
  //   FILE* fOut = fopen("gmmHistModel.bin", "wb");
  //   unsigned int ui[6];
  //   ui[0] = vdims[0]; ui[1] = vdims[1]; ui[2] = vdims[2]; //raw xyz
  //   ui[3] = blkSize; //block size;
  //   ui[4] = blkStIndex.size(); //number of blocks
  //   ui[5] = binidAry.GetNumberOfValues(); //number of non-zeros bins
  //   fwrite (ui , sizeof(unsigned int), 6, fOut);

  //   fwrite (&blkStIndex[0], sizeof(vtkm::Id), blkStIndex.size(), fOut);
  //   fwrite (&blkLensVec[0], sizeof(std::vector<vtkm::Id>::value_type), blkLensVec.size(), fOut);

  //   std::vector<vtkm::Id> binVec;
  //   std::vector<vtkm::Id> freqVec;
  //   std::vector<float> weights;
  //   std::vector<float> means;
  //   std::vector<float> covs;
  //   for( int i=0; i<binidAry.GetNumberOfValues(); i++ ){
  //     binVec.push_back(binidAry.GetPortalControl().Get(i));
  //     freqVec.push_back(freqAry.GetPortalControl().Get(i));
  //     vtkm::worklet::GMM<4> gmm = em.gmmsHandle.GetPortalControl().Get(i);
  //     for( int c=0; c<4; c++ ){
  //       //weight
  //       weights.push_back(gmm.weights[c]);
  //       means.push_back(gmm.means[c][0]); means.push_back(gmm.means[c][1]); means.push_back(gmm.means[c][2]);
  //       covs.push_back(gmm.covMats[c][0][0]); covs.push_back(gmm.covMats[c][0][1]); covs.push_back(gmm.covMats[c][0][2]); 
  //       covs.push_back(gmm.covMats[c][1][1]); covs.push_back(gmm.covMats[c][1][2]);
  //       covs.push_back(gmm.covMats[c][2][2]);
  //     }
  //   }
  //   fwrite (&binVec[0], sizeof(std::vector<vtkm::Id>::value_type), binVec.size(), fOut);
  //   fwrite (&freqVec[0], sizeof(std::vector<vtkm::Id>::value_type), freqVec.size(), fOut);
  //   fwrite (&weights[0], sizeof(std::vector<float>::value_type), weights.size(), fOut);
  //   fwrite (&means[0], sizeof(std::vector<float>::value_type), means.size(), fOut);
  //   fwrite (&covs[0], sizeof(std::vector<float>::value_type), covs.size(), fOut);

  //   fclose(fOut);
  // }
  // printf("finish file write\n");

  // //// GMM Training 
  // // prepare GMM traiing data , should be done by Group By statement
  // vtkm::cont::Timer<> gmmSamplePreprocessTimer;
  // vtkm::worklet::VtkmTable vtkmTable;
  // vtkmTable.From(ds);
  // vtkmTable.Select(std::vector<std::string>{"0,16(X)", "0,16(Y)","0,16(Z)", "128(V)", "X", "Y", "Z"  });
  // vtkmTable.SortBy(std::vector<std::string>{"0,16(X)", "0,16(Y)","0,16(Z)", "128(V)"});
  // vtkm::cont::DataSet resultDS = vtkmTable.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
  
  // std::vector<vtkm::Id> gmmIds;
  // std::vector< vtkm::worklet::PointND > trainData;

  // vtkm::cont::ArrayHandle<vtkm::Id> bxAry; resultDS.GetField(0).GetData().CopyTo(bxAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> byAry; resultDS.GetField(1).GetData().CopyTo(byAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> bzAry; resultDS.GetField(2).GetData().CopyTo(bzAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> binAry; resultDS.GetField(3).GetData().CopyTo(binAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> xAry; resultDS.GetField(4).GetData().CopyTo(xAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> yAry; resultDS.GetField(5).GetData().CopyTo(yAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> zAry; resultDS.GetField(6).GetData().CopyTo(zAry);

  // vtkm::Id gmmCnt = 0;
  // vtkm::Id pbx = bxAry.GetPortalConstControl().Get(0);
  // vtkm::Id pby = byAry.GetPortalConstControl().Get(0);
  // vtkm::Id pbz = bzAry.GetPortalConstControl().Get(0);
  // vtkm::Id pbin = binAry.GetPortalConstControl().Get(0);
  // for( int i=0; i< bxAry.GetNumberOfValues(); i++ ){
  //   vtkm::Id bx = bxAry.GetPortalConstControl().Get(i);
  //   vtkm::Id by = byAry.GetPortalConstControl().Get(i);
  //   vtkm::Id bz = bzAry.GetPortalConstControl().Get(i);
  //   vtkm::Id bin = binAry.GetPortalConstControl().Get(i);
  //   vtkm::Id x = xAry.GetPortalConstControl().Get(i);
  //   vtkm::Id y = yAry.GetPortalConstControl().Get(i);
  //   vtkm::Id z = zAry.GetPortalConstControl().Get(i);
  //   if( bx != pbx || by != pby || bz != pbz || bin != pbin ){
  //     pbx = bx;
  //     pby = by;
  //     pbz = bz;
  //     pbin = bin;
  //     gmmCnt ++;
  //   }
  //   vtkm::Vec<vtkm::Float32,3> coordi3D;
  //   coordi3D[0] = x;
  //   coordi3D[1] = y;
  //   coordi3D[2] = z;

  //   gmmIds.push_back( gmmCnt );
  //   trainData.push_back(coordi3D);
  // }
  // gmmCnt ++;
  // std::cout<< "GMM Sample Prepare Time: " << gmmSamplePreprocessTimer.GetElapsedTime() << std::endl;
  // std::cout << "Total GMMs: " << gmmCnt << std::endl;
  
  // vtkm::cont::Timer<> gmmTimer;
  // vtkm::worklet::GMMTraining<4> em;
  // em.Run(trainData, gmmIds, gmmCnt, 10, VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
  // std::cout<< "GMM EM Time: " << gmmTimer.GetElapsedTime() << std::endl;

  // //Compute Histogram 
  // // From
  // vtkm::cont::Timer<> histConstructTimer;
  // vtkm::worklet::VtkmTable vtkmTableHist;
  // vtkmTableHist.From(ds);
  // vtkmTableHist.Select( std::vector<std::string>{"0,16(X)", "0,16(Y)","0,16(Z)", "128(V)", "COUNT(V)"} );
  // vtkmTableHist.GroupBy(std::vector<std::string>{"0,16(X)", "0,16(Y)","0,16(Z)", "128(V)"});
  // vtkmTableHist.SortBy(std::vector<std::string>{"0,16(X)", "0,16(Y)","0,16(Z)", "128(V)"});
  // vtkm::cont::DataSet histDS = vtkmTableHist.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
  // std::cout<< "Histogram Construct Time: " << histConstructTimer.GetElapsedTime() << std::endl;
  

  // vtkm::cont::Timer<> gmmHistReconstructTimer;
  // vtkm::cont::ArrayHandle<vtkm::Id> blkxAry; histDS.GetField(0).GetData().CopyTo(blkxAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> blkyAry; histDS.GetField(1).GetData().CopyTo(blkyAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> blkzAry; histDS.GetField(2).GetData().CopyTo(blkzAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> binidAry; histDS.GetField(3).GetData().CopyTo(binidAry);
  // vtkm::cont::ArrayHandle<vtkm::Id> freqAry; histDS.GetField(4).GetData().CopyTo(freqAry);
  // // compute the table QX, QY, QZ "StartIdx, Lens"
  // vtkm::Id blkCnt = 0;
  // vtkm::cont::DataSet indexDs;
  // indexDs.AddField(vtkm::cont::Field("BlkX", vtkm::cont::Field::ASSOC_POINTS, blkxAry));
  // indexDs.AddField(vtkm::cont::Field("BlkY", vtkm::cont::Field::ASSOC_POINTS, blkyAry));
  // indexDs.AddField(vtkm::cont::Field("BlkZ", vtkm::cont::Field::ASSOC_POINTS, blkzAry));
  // vtkm::worklet::VtkmTable vtkmTableIndex;
  // vtkmTableIndex.From(indexDs);
  // vtkmTableIndex.Select( std::vector<std::string>{"BlkX", "BlkY","BlkZ", "COUNT(BlkX)"} );
  // vtkmTableIndex.GroupBy(std::vector<std::string>{"BlkX", "BlkY","BlkZ"});
  // vtkmTableIndex.SortBy(std::vector<std::string>{"BlkX", "BlkY","BlkZ"});
  // vtkm::cont::DataSet resultIndexDs = vtkmTableIndex.Query(VTKM_DEFAULT_DEVICE_ADAPTER_TAG());
  // std::vector<vtkm::Id> blkStIndex;
  // vtkm::cont::ArrayHandle<vtkm::Id> riBlkX; resultIndexDs.GetField(0).GetData().CopyTo(riBlkX);
  // vtkm::cont::ArrayHandle<vtkm::Id> riBlkY; resultIndexDs.GetField(1).GetData().CopyTo(riBlkY);
  // vtkm::cont::ArrayHandle<vtkm::Id> riBlkZ; resultIndexDs.GetField(2).GetData().CopyTo(riBlkZ);
  // vtkm::cont::ArrayHandle<vtkm::Id> blkLens; resultIndexDs.GetField(3).GetData().CopyTo(blkLens);
  // vtkm::Id countSum = 0;
  // for( int i=0; i<blkLens.GetNumberOfValues(); i++ ){
  //   blkStIndex.push_back(countSum);
  //   if( i != blkLens.GetNumberOfValues()-1 ) //do not do this at the last one element
  //     countSum += blkLens.GetPortalConstControl().Get(i);
  // }


  // {
  //   // output our data structure to file
  //   FILE* fOut = fopen("gmmHistModel.bin", "wb");
  //   unsigned int ui[6];
  //   ui[0] = vdims[0]; ui[1] = vdims[1]; ui[2] = vdims[2]; //raw xyz
  //   ui[3] = blkSize; //block size;
  //   ui[4] = blkStIndex.size(); //number of blocks
  //   ui[5] = binidAry.GetNumberOfValues(); //number of non-zeros bins
  //   fwrite (ui , sizeof(unsigned int), 6, fOut);

  //   fwrite (&blkStIndex[0], sizeof(vtkm::Id), blkStIndex.size(), fOut);
  //   std::vector<vtkm::Id> blkLensVec;
  //   for( int i=0; i<blkLens.GetNumberOfValues(); i++ )blkLensVec.push_back(blkLens.GetPortalConstControl().Get(i));
  //   fwrite (&blkLensVec[0], sizeof(std::vector<vtkm::Id>::value_type), blkLensVec.size(), fOut);

  //   std::vector<vtkm::Id> binVec;
  //   std::vector<vtkm::Id> freqVec;
  //   std::vector<float> weights;
  //   std::vector<float> means;
  //   std::vector<float> covs;
  //   for( int i=0; i<binidAry.GetNumberOfValues(); i++ ){
  //     binVec.push_back(binidAry.GetPortalControl().Get(i));
  //     freqVec.push_back(freqAry.GetPortalControl().Get(i));
  //     vtkm::worklet::GMM<4> gmm = em.gmmsHandle.GetPortalControl().Get(i);
  //     for( int c=0; c<4; c++ ){
  //       //weight
  //       weights.push_back(gmm.weights[c]);
  //       means.push_back(gmm.means[c][0]); means.push_back(gmm.means[c][1]); means.push_back(gmm.means[c][2]);
  //       covs.push_back(gmm.covMats[c][0][0]); covs.push_back(gmm.covMats[c][0][1]); covs.push_back(gmm.covMats[c][0][2]); 
  //       covs.push_back(gmm.covMats[c][1][1]); covs.push_back(gmm.covMats[c][1][2]);
  //       covs.push_back(gmm.covMats[c][2][2]);
  //     }
  //   }
  //   fwrite (&binVec[0], sizeof(std::vector<vtkm::Id>::value_type), binVec.size(), fOut);
  //   fwrite (&freqVec[0], sizeof(std::vector<vtkm::Id>::value_type), freqVec.size(), fOut);
  //   fwrite (&weights[0], sizeof(std::vector<float>::value_type), weights.size(), fOut);
  //   fwrite (&means[0], sizeof(std::vector<float>::value_type), means.size(), fOut);
  //   fwrite (&covs[0], sizeof(std::vector<float>::value_type), covs.size(), fOut);

  //   fclose(fOut);
  // }
  // printf("finish file write\n");

} // TestFieldHistogram
}//namespace

int UnitTestGMMTraining(int, char* [])
{
  return vtkm::cont::testing::Testing::Run(TestGMMTraining);
}
