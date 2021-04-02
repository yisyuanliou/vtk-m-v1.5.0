//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/worklet/VolumeRendering.h>

#include <vtkm/cont/DataSet.h>
#include <vtkm/worklet/GMMTraining.h>
#include <vtkm/cont/testing/Testing.h>

#include <fstream>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib> 
#include <ctime>

namespace
{
typedef vtkm::Float64 Real;
const int VARs = 3;
const int nGauComps = 4;
//
// Make a simple 2D, 1000 point dataset populated with stat distributions
//
vtkm::cont::DataSet ReadTestDataSet()
{
  vtkm::cont::DataSet dataSet;

  const int dimension = 3;
  const int xVerts = 500;
  const int yVerts = 500;
  const int zVerts = 100;
  const int nVerts = xVerts * yVerts * zVerts;

  const int xCells = xVerts - 1;
  const int yCells = yVerts - 1;
  const int zCells = zVerts - 1;
  const int nCells = xCells * yCells * zCells;

  // vtkm::Float32 data[nVerts] = {
  //   1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
  //   33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64
  // };


  vtkm::Float32* data = (vtkm::Float32*)malloc(nVerts * sizeof(vtkm::Float32));
  std::ifstream fileIn("aaa.bin", std::ios::binary);
  float f;
  int i = 0;
  while (fileIn.read(reinterpret_cast<char*>(&f), sizeof(float)))
  {
    data[i++] = f;
  }

  vtkm::cont::ArrayHandleUniformPointCoordinates coordinates(vtkm::Id3(xVerts, yVerts, zVerts));
  dataSet.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coordinates", coordinates));

  // Set point scalars
  dataSet.AddField(vtkm::cont::make_Field(
    "p_data", vtkm::cont::Field::Association::POINTS, data, nVerts, vtkm::CopyFlag::On));

  // Set cell scalars
  dataSet.AddField(vtkm::cont::make_Field(
    "c_data", vtkm::cont::Field::Association::CELL_SET, data, nCells, vtkm::CopyFlag::On));

  vtkm::cont::CellSetStructured<dimension> cellSet;

  //Set regular structure
  cellSet.SetPointDimensions(vtkm::make_Vec(xVerts, yVerts, zVerts));
  dataSet.SetCellSet(cellSet);
  //free(data);
  return dataSet;
}

vtkm::Id3 dims;
std::vector<float> plvPvVec;
std::vector<int> blkIdxVec;
int D0, D1, D2;
int bins;
int blkSize;
int blkD0, blkD1, blkD2;

std::vector<vtkm::Float64> aryDataOld;
std::vector<vtkm::Id> aryOffsetOld;

std::vector<vtkm::Float64> aryData;
std::vector<vtkm::Id> aryOffset;
vtkm::worklet::GMMTraining<nGauComps, VARs, Real> em;
int *bin2gmmidx;

int maxLen = 0;


void loadPLVnPV(){
  ////***** new data loader 
  vtkm::Id3 vdims;
  vtkm::Id blkSize;
  vtkm::Id nBlocks;
  vtkm::Id nonZeroBins;
  
  int bins = 128;
  int* blkStIndex;
  int* blkLens;
  int* binidAry;
  int* freqAry;
  
  
  // load file
  {
    printf("loading\n");
    FILE *fIn = fopen("./bin/data/GMMHistModel.bin", "rb");
    if (fIn == NULL)
	  {
		  std::cout << "file open failed!" << std::endl;
	  }

    unsigned int ui[6];
    fread (ui,sizeof(int),6,fIn);
    vdims[0] = ui[0]; vdims[1] = ui[1]; vdims[2] = ui[2];
    blkSize = ui[3];
    nBlocks = ui[4];
    nonZeroBins = ui[5];
    printf("%d %d %d %d %d %d\n", vdims[0], vdims[1], vdims[2], ui[3], ui[4], ui[5]);

    blkStIndex = (int*)malloc(sizeof(int) * nBlocks);
    blkLens = (int*)malloc(sizeof(int) * nBlocks);
    fread (blkStIndex,sizeof(int), nBlocks,fIn );
    fread (blkLens,sizeof(int), nBlocks, fIn );

    binidAry = (int*)malloc(sizeof(int) * nonZeroBins);
    freqAry = (int*)malloc(sizeof(int) * nonZeroBins);
    bin2gmmidx = (int*)malloc(sizeof(int) * bins * nBlocks);
    fread (binidAry, sizeof(int), nonZeroBins, fIn );
    fread (freqAry, sizeof(int), nonZeroBins, fIn );
    fread (bin2gmmidx, sizeof(int), bins * nBlocks, fIn );

    /*weights = (float*)malloc(sizeof(float) * nonZeroBins * 4); 
    means = (float*)malloc(sizeof(float) * nonZeroBins * 4 * 3); 
    covs = (float*)malloc(sizeof(float) * nonZeroBins * 4 * 6 ); 
    fread (weights, sizeof(float), nonZeroBins * 4, fIn );
    fread (means, sizeof(float), nonZeroBins * 4 * 3, fIn );
    fread (covs, sizeof(float), nonZeroBins * 4 * 6, fIn );*/

    /*em.gmmsHandle.Allocate(nonZeroBins);
    printf("loading\n");
    printf("%f %f %f\n", weights, means, covs);
    for( int i=0; i<nonZeroBins; i++ ){

      weights = (float*)malloc(sizeof(float) * 4); 
      means = (float*)malloc(sizeof(float) * 4 * 3); 
      covs = (float*)malloc(sizeof(float) * 4 * 6 );
      fread (weights, sizeof(float), 4, fIn );
      fread (means, sizeof(float), 4*3, fIn );
      fread (covs, sizeof(float), 4*6, fIn );
      vtkm::worklet::GMM<4> gmm;
      
      //gmm.setGMM(weights + i * 4, means + i * 4 * 3, covs + i * 4 * 6);
      gmm.setGMM(weights, means, covs);
      em.gmmsHandle.GetPortalControl().Set(i, gmm);
    }*/

    fclose(fIn);
  }

  printf("non zero bins %d\n", nonZeroBins);

  printf("finish loading\n");
  ////***** end - new data loader

  //// **** transfer model data into vtkm array
  D0 = vdims[0]; D1 = vdims[1]; D2 = vdims[2]; 
  blkD0 = static_cast<int>(ceil(float(D0) / float(blkSize)));
  blkD1 = static_cast<int>(ceil(float(D1) / float(blkSize)));
  blkD2 = static_cast<int>(ceil(float(D2) / float(blkSize)));

  std::cout << "blkD0123  " << blkD0 << " "<< blkD1 << " "<< blkD2 << std::endl;
  dims[0] = blkD2;//coordinate trick here
  dims[1] = blkD1;
  dims[2] = blkD0;

  int numBlocks = 0;
  int nzBinsCnt = 0;
  for (int blkd0 = 0; blkd0 < blkD0; blkd0++){
    for (int blkd1 = 0; blkd1 < blkD1; blkd1++){
      for (int blkd2 = 0; blkd2 < blkD2; blkd2++){
        int blkLen = blkLens[numBlocks]; 
        numBlocks++;
        // printf("num %d\n", numBlocks);
        aryOffsetOld.push_back(aryDataOld.size());
        
        //put meta data
        // aryDataOld.push_back(float(bins));
        // aryDataOld.push_back(float(blkSize));
        // aryDataOld.push_back(float(D0));
        // aryDataOld.push_back(float(D1));
        // aryDataOld.push_back(float(D2));

        //put block ID here
        // aryDataOld.push_back(float(blkd0));
        // aryDataOld.push_back(float(blkd1));
        // aryDataOld.push_back(float(blkd2));
        int blkcnt = blkd2*blkD0*blkD1 + blkd1*blkD0+blkd0;

        int nzBinCntInBlk = 0;
        int blkfreqsum = 0;
        
        for (int b = 0; b < bins; b++){
          //printf("bid %d %d %d \n", b, binidAry[nzBinsCnt], nzBinsCnt);
          if(b != binidAry[nzBinsCnt]||blkfreqsum>=blkSize*blkSize*blkSize){ //freq==0
            aryDataOld.push_back(0);
          }
          else{//freq != 0
            //printf("freq %d \n", freqAry[nzBinsCnt]);
            aryDataOld.push_back(freqAry[nzBinsCnt]);
            blkfreqsum += freqAry[nzBinsCnt];
            // printf("%d\n", blkfreqsum);
            nzBinCntInBlk ++; 

            aryDataOld.push_back(4);//current version we have fix 4 gaussian component

            //int blkcnt = blkd2*blkD0*blkD1 + blkd1*blkD0+blkd0;//blkd0*blkD2*blkD1 + blkd1*blkD2+blkd2;
            int gmmid = bin2gmmidx[(blkcnt*bins)+b];
            aryDataOld.push_back(static_cast<vtkm::Float64>(gmmid));
            //printf("numblock %d %d %d %d\n", numBlocks, blkcnt, b, gmmid);
            /*
            vtkm::worklet::GMM<nGauComps, VARs, Real> gmm = em.gmmsHandle.GetPortalControl().Get(gmmid);

            for (int g = 0; g < 4; g++){
              
              aryDataOld.push_back(gmm.weights[g]);
              aryDataOld.push_back(gmm.means[g][0]);
              aryDataOld.push_back(gmm.means[g][1]);
              aryDataOld.push_back(gmm.means[g][2]);
              printf("%f %f %f %f\n", gmm.weights[g], gmm.means[g][0], gmm.means[g][1], gmm.means[g][2]);

              aryDataOld.push_back(gmm.covMats[g][0][0]);
              aryDataOld.push_back(gmm.covMats[g][0][1]);
              aryDataOld.push_back(gmm.covMats[g][0][2]);
              aryDataOld.push_back(gmm.covMats[g][1][0]);
              aryDataOld.push_back(gmm.covMats[g][1][1]);
              aryDataOld.push_back(gmm.covMats[g][1][2]);
              aryDataOld.push_back(gmm.covMats[g][2][0]);
              aryDataOld.push_back(gmm.covMats[g][2][1]);
              aryDataOld.push_back(gmm.covMats[g][2][2]);
              // printf("%f %f %f\n%f %f %f\n%f %f %f\n\n", gmm.covMats[g][0][0], gmm.covMats[g][0][1], gmm.covMats[g][0][2], gmm.covMats[g][1][0], gmm.covMats[g][1][1], gmm.covMats[g][1][2], gmm.covMats[g][2][0], gmm.covMats[g][2][1], gmm.covMats[g][2][2]);

              aryDataOld.push_back(gmm.logPDet[g]);

              aryDataOld.push_back(gmm.precCholMat[g][0][0]);
              aryDataOld.push_back(gmm.precCholMat[g][0][1]);
              aryDataOld.push_back(gmm.precCholMat[g][0][2]);
              aryDataOld.push_back(gmm.precCholMat[g][1][0]);
              aryDataOld.push_back(gmm.precCholMat[g][1][1]);
              aryDataOld.push_back(gmm.precCholMat[g][1][2]);
              aryDataOld.push_back(gmm.precCholMat[g][2][0]);
              aryDataOld.push_back(gmm.precCholMat[g][2][1]);
              aryDataOld.push_back(gmm.precCholMat[g][2][2]);
              // printf("%f %f %f\n%f %f %f\n%f %f %f\n\n", gmm.precCholMat[g][0][0], gmm.precCholMat[g][0][1], gmm.precCholMat[g][0][2], gmm.precCholMat[g][1][0], gmm.precCholMat[g][1][1], gmm.precCholMat[g][1][2], gmm.precCholMat[g][2][0], gmm.precCholMat[g][2][1], gmm.precCholMat[g][2][2]);

              aryDataOld.push_back(gmm.lowerMat[g][0][0]);
              aryDataOld.push_back(gmm.lowerMat[g][0][1]);
              aryDataOld.push_back(gmm.lowerMat[g][0][2]);
              aryDataOld.push_back(gmm.lowerMat[g][1][0]);
              aryDataOld.push_back(gmm.lowerMat[g][1][1]);
              aryDataOld.push_back(gmm.lowerMat[g][1][2]);
              aryDataOld.push_back(gmm.lowerMat[g][2][0]);
              aryDataOld.push_back(gmm.lowerMat[g][2][1]);
              aryDataOld.push_back(gmm.lowerMat[g][2][2]);
              // printf("%f %f %f\n%f %f %f\n%f %f %f\n", gmm.lowerMat[g][0][0], gmm.lowerMat[g][0][1], gmm.lowerMat[g][0][2], gmm.lowerMat[g][1][0], gmm.lowerMat[g][1][1], gmm.lowerMat[g][1][2], gmm.lowerMat[g][2][0], gmm.lowerMat[g][2][1], gmm.lowerMat[g][2][2]);
            }//for g*/
            nzBinsCnt ++;
          }//else
        }//b
        //printf("b %d\n", nzBinCntInBlk);

      }
    }
  }
  printf("nzbinCnt %d \n", nzBinsCnt);


  for( unsigned int i=0; i<aryDataOld.size(); i++ ){
    if( aryDataOld[i]!=aryDataOld[i] || aryDataOld[i]>999999.0f || aryDataOld[i]<-999999.0f ) {
      printf("Load check %d %f\n", i, aryDataOld[i]);
      aryDataOld[i] = 0;
    }
  }

  int vtkmD0 = blkD0;
  int vtkmD1 = blkD1;
  int vtkmD2 = blkD2;
  for (int blkd0 = 0; blkd0 < vtkmD0; blkd0++){
    for (int blkd1 = 0; blkd1 < vtkmD1; blkd1++){
      for (int blkd2 = 0; blkd2 < vtkmD2; blkd2++){
        aryOffset.push_back(aryData.size());

        int oldIdx = ( blkd0 * blkD1 * blkD2 ) + ( blkd1 * blkD2 ) + blkd2;
        int stAryIdx = aryOffsetOld[oldIdx];
        int utAryIdx = ( oldIdx == (blkd0*blkD1*blkD2 - 1) ) ? aryDataOld.size() : aryOffsetOld[oldIdx+1];
        // printf("%d %d %d\n", oldIdx, stAryIdx, utAryIdx);

        for( int i = stAryIdx; i < utAryIdx; i++ ){
          aryData.push_back(aryDataOld[i]);
        }
      }
    }
  }
  aryOffset.push_back(aryData.size());

  printf("finish LVnPVL\n");
}



//
// Create a dataset with known point data and cell data (statistical distributions)
// Extract arrays of point and cell fields
// Create output structure to hold histogram bins
// Run FieldHistogram filter
//
void TestVolumeRendering()
{
  vtkm::Float32 x = 500;
  vtkm::Float32 y = 500;
  vtkm::Float32 z = 100;
  vtkm::Float32 angle = 45;
  auto boxSize = vtkm::make_Vec(x, y, z);
  vtkm::Float32 xResolution = 1000;
  vtkm::Float32 yResolution = 1000;
  auto imageResolution = vtkm::make_Vec(xResolution, yResolution);
  vtkm::Float32 xRotation = vtkm::Pi();
  vtkm::Float32 yRotation = vtkm::Pi();
  vtkm::Float32 zRotation = vtkm::Pi_4();
  auto rotation = vtkm::make_Vec(xRotation, yRotation, zRotation);
  vtkm::cont::ArrayHandle<vtkm::Float32> image;
  image.Allocate(xResolution * yResolution);


  loadPLVnPV();

  vtkm::cont::ArrayHandle<vtkm::Float64> sourceArray = vtkm::cont::make_ArrayHandle(aryData);
  vtkm::cont::ArrayHandle<vtkm::Id> offsetArray = vtkm::cont::make_ArrayHandle(aryOffset);

  vtkm::worklet::VolumeRendering vr;
  vr.Run(boxSize, imageResolution, rotation, angle, image, offsetArray, sourceArray);
  
} // TestVolumeRendering
}

int UnitTestVolumeRendering(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(TestVolumeRendering, argc, argv);
}
