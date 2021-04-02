//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#ifndef vtk_m_worklet_VolumeRendering_h
#define vtk_m_worklet_VolumeRendering_h

#include <vtkm/imageIO.h>
#include <vtkm/Math.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayGetValues.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/Field.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/Matrix.h>
#include <vtkm/worklet/GMMTraining.h>
#include <cstdio>

namespace
{
// GCC creates false positive warnings for signed/unsigned char* operations.
// This occurs because the values are implicitly casted up to int's for the
// operation, and than  casted back down to char's when return.
// This causes a false positive warning, even when the values is within
// the value types range
#if defined(VTKM_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif // gcc
#if defined(VTKM_GCC)
#pragma GCC diagnostic pop
#endif // gcc
}

namespace vtkm
{
namespace worklet
{

//simple functor that prints basic statistics
class VolumeRendering
{
public:
  
  // Calculate the adjacent difference between values in ArrayHandle
  class Intersect : public vtkm::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn inputIndex, FieldOut checkHit, FieldOut xyz);
    using ExecutionSignature = void(_1, _2,_3);
    using InputDomain = _1;

    vtkm::Vec<vtkm::Vec<vtkm::Float32, 3>, 2> bounds;
    vtkm::Vec<vtkm::Float32, 3> cameraLoc;
    vtkm::Vec<vtkm::Float32, 3> imageUpperLeftLoc;
    vtkm::Vec<vtkm::Id, 2> imageResolution;
    vtkm::Vec<vtkm::Float32, 3> horizontal;
    vtkm::Vec<vtkm::Float32, 3> vertical;

    VTKM_CONT
    Intersect(vtkm::Vec<vtkm::Float32, 3> boxSize0,
              vtkm::Vec<vtkm::Float32, 3> cameraLoc0,
              vtkm::Vec<vtkm::Float32, 3> imageUpperLeftLoc0,
              vtkm::Vec<vtkm::Id, 2> imageResolution0,
              vtkm::Vec<vtkm::Float32, 3> horizontal0, 
              vtkm::Vec<vtkm::Float32, 3> vertical0)
      : cameraLoc(cameraLoc0)
      , imageUpperLeftLoc(imageUpperLeftLoc0)
      , imageResolution(imageResolution0)
      , horizontal(horizontal0)
      , vertical(vertical0)
    { 
      bounds[0] = -boxSize0 / 2;
      bounds[1] = boxSize0 / 2;
      //std::cout << imageCenter[0] << " " << imageCenter[1] << " " << imageCenter[2] << "\n";
    }

    VTKM_EXEC void operator()(const vtkm::Id& inputIndex,
                              vtkm::Float32& checkHit,
                              vtkm::Vec<vtkm::Float32, 3>& xyz) const
    {
      vtkm::Id x = inputIndex % imageResolution[0];
      vtkm::Id y = (inputIndex - x) / imageResolution[0];
      vtkm::Vec<vtkm::Float32, 3> pixelLoc = imageUpperLeftLoc - x * horizontal - y * vertical;
      //std::cout << pixelLoc << "\n";
      vtkm::Vec<vtkm::Float32, 3> ray = pixelLoc - cameraLoc;
      vtkm::Normalize(ray);
      ray = vtkm::make_Vec(1.0, 1.0, 1.0) / ray;
      vtkm::Vec<vtkm::Int16, 3> sign = ray < 0;

      vtkm::Vec<vtkm::Float32, 3> tmin, tmax;
      tmin[0] = bounds[sign[0]][0];
      tmin[1] = bounds[sign[1]][1];
      tmin[2] = bounds[sign[2]][2];
      tmax[0] = bounds[1 - sign[0]][0];
      tmax[1] = bounds[1 - sign[1]][1];
      tmax[2] = bounds[1 - sign[2]][2];
      tmin = (tmin - cameraLoc) * ray;
      tmax = (tmax - cameraLoc) * ray;
      vtkm::Float32 minT = vtkm::Max(vtkm::Max(vtkm::Min(tmin[0], tmax[0]), vtkm::Min(tmin[1], tmax[1])),vtkm::Min(tmin[2], tmax[2]));
      vtkm::Float32 maxT = vtkm::Min(vtkm::Min(vtkm::Max(tmin[0], tmax[0]), vtkm::Max(tmin[1], tmax[1])),vtkm::Max(tmin[2], tmax[2]));

      // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
      if (maxT < 0)
      {
        checkHit = -1;
      }
      // if tmin > tmax, ray doesn't intersect AABB
      else if ((minT > maxT) && (minT - maxT) > 0.001)
      {
        //if (y == 38)
        //    std::cout << x << ": " << minT - maxT << "\n";
        checkHit = -1;
      }
      else
      {
        checkHit = minT;
        auto err = 0.0001;
        xyz = minT * vtkm::make_Vec(1.0, 1.0, 1.0) / ray + cameraLoc;
        //std::cout<<"xyz: "<<xyz[0]<<" "<<xyz[1]<<" "<<xyz[2]<<" "<<minT<<std::endl;
        ///////////////
        if (vtkm::Abs(xyz[0] - bounds[0][0]) < err)
          checkHit = 1;
        else if (vtkm::Abs(xyz[0] - bounds[1][0]) < err)
          checkHit = 2;
        else if (vtkm::Abs(xyz[1] - bounds[0][1]) < err)
          checkHit = 3;
        else if (vtkm::Abs(xyz[1] - bounds[1][1]) < err)
          checkHit = 4;
        else if (vtkm::Abs(xyz[2] - bounds[0][2]) < err)
          checkHit = 5;
        else
          checkHit = 6;
        //////////////
      }

    }
  };

  template<typename FieldType, typename Storage>
  void Output(vtkm::Vec<vtkm::Float32, 3> boxSize,
              vtkm::Vec<vtkm::Float32, 3> cameraLoc0,
              vtkm::cont::ArrayHandle<FieldType, Storage> arr,
              vtkm::Vec<vtkm::Id, 2> imageResolution, 
              vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3> >& xyz,
              vtkm::cont::ArrayHandle<vtkm::Float64> &aryData,
              vtkm::cont::ArrayHandle<vtkm::Id> &aryOffset)
  {
    //// Load GMM from a file, programmer has to know the nGauComps, VARs, Real used to train the GMM
    char gmmFilePath[20]="./bin/data/gmm.bin";
    vtkm::worklet::GMMTraining<4, 3, vtkm::Float64> em;
	  em.LoadGMMsFile(gmmFilePath);

    std::default_random_engine rng;
    std::uniform_real_distribution<vtkm::Float32> dr(0.0f, 1.0f);
    ColorImage img;
    img.init(imageResolution[0], imageResolution[1]);

    const vtkm::Float32 numberOfBins = 128;
    const vtkm::Float32 fieldMinValue = -4900;
    const vtkm::Float32 fieldMaxValue = 2400;
    vtkm::Float32 delta = (fieldMaxValue-fieldMinValue)/numberOfBins;
    int test = 1; // 1: GMM, 2: raw data, 3: test

    FILE* fp = fopen("./pf16.bin", "rb");
    if (fp == NULL)
    {
      std::cout << "file open failed!" << std::endl;
    }
    float value;
    int index= 0;
    vtkm::Float32 *gtData = (vtkm::Float32 *)malloc(500*500*100 * sizeof(vtkm::Float32));
    for (int d0 = 0; d0 < 500; d0++) {
      for (int d1 = 0; d1 < 500; d1++) {
        for (int d2 = 0; d2 < 100; d2++) {
          int tmp = fread(&value, sizeof(float), 1, fp);
          gtData[index] = value;
          index += 1;
        }
      }
    }
    // std::cout<<"index "<<index<<std::endl;
    printf("%d\n", arr.GetNumberOfValues());
    for (int i = 0; i < arr.GetNumberOfValues(); i++)
    {
      if (arr.GetPortalConstControl().Get(i) != -1)
      {
        int x = i % imageResolution[0];
        int y = (i - x) / imageResolution[0];
        float R=0;
        float G=0;
        float B=0;
        float total_alpha=0;
        if(test==1){ // GMM rendering
          int xx = static_cast<int>(xyz.GetPortalConstControl().Get(i)[0]+ boxSize[0]/2);
          int yy = static_cast<int>(xyz.GetPortalConstControl().Get(i)[1]+ boxSize[1]/2);
          int zz = static_cast<int>(xyz.GetPortalConstControl().Get(i)[2]+ boxSize[2]/2);
          vtkm::Vec<Float64, 3> xyz_ = vtkm::make_Vec(xx, yy, zz);

          vtkm::Vec<Float64, 3> dir = xyz_ - cameraLoc0;

          do
          {
            if(xx>=boxSize[0]) xx-=1;
            if(yy>=boxSize[1]) yy-=1;
            if(zz>=boxSize[2]) zz-=1;
            // std::cout<<xx<<" "<<yy<<" "<<zz<<" "<<xx*boxSize[2]*boxSize[1]+yy*boxSize[2]+zz<<std::endl;
            float sample = gtData[(int)(zz*boxSize[0]*boxSize[1]+yy*boxSize[0]+xx)];
            vtkm::Vec<vtkm::Float32, 4> RGBD{0};
            transfer_func(RGBD, sample);
            float w = (1-total_alpha)*RGBD[3];
            total_alpha+=w;
            R += (RGBD[0]*255)*w;
            G += (RGBD[1]*255)*w;//0;
            B += (RGBD[2]*255)*w;//0;
            if(total_alpha>=0.99) break;
            
            // std::cout<<"rgbd "<<total_alpha<<" "<<RGBD[0]*255<<" "<<RGBD[1]*255<<" "<<RGBD[2]*255<<std::endl;
            xx += static_cast<int>(0.01*dir[0]);
            yy += static_cast<int>(0.01*dir[1]);
            zz += static_cast<int>(0.01*dir[2]);
            xyz_ = vtkm::make_Vec(xx, yy, zz);

          }while(xx <= boxSize[0] && yy <= boxSize[1] && zz<=boxSize[2]);

        }

        else if(test==2){  //rae data rendering
          int xx = static_cast<int>(xyz.GetPortalConstControl().Get(i)[0]+ boxSize[0]/2);
          int yy = static_cast<int>(xyz.GetPortalConstControl().Get(i)[1]+ boxSize[1]/2);
          int zz = static_cast<int>(xyz.GetPortalConstControl().Get(i)[2]+ boxSize[2]/2);
          vtkm::Vec<Float64, 3> xyz_ = vtkm::make_Vec(xx, yy, zz);

          vtkm::Vec<Float64, 3> dir = xyz_ - cameraLoc0;

          do
          {
            int blkxx = static_cast<int>((xx -1)/20);
            int blkyy = static_cast<int>((yy -1)/20);
            int blkzz = static_cast<int>((zz -1)/20);

            int blkindex = blkxx*5*25 + blkyy*5 + blkzz;
            // printf("pos %f, %f %f \n",xyz.GetPortalConstControl().Get(i)[0], xyz.GetPortalConstControl().Get(i)[1], xyz.GetPortalConstControl().Get(i)[2]);

            //printf("%d %d %d %d\n", blkxx, blkyy, blkzz, blkindex);

            int number = aryOffset.GetPortalConstControl().Get(blkindex+1)-aryOffset.GetPortalConstControl().Get(blkindex);
            //printf("num %d %d\n", number, aryOffset.GetPortalConstControl().Get(blkindex));
            vtkm::Float32 postProb[(int)numberOfBins];
            float postProbSum = 0.0f;
            int bincnt = 0;

            for (int k = (int)aryOffset.GetPortalConstControl().Get(blkindex); k < (int)aryOffset.GetPortalConstControl().Get(blkindex+1); ) {
                //std::cout << stIdx << " d " << BinsToGMMId[stIdx*numberOfBins + k] << std::endl;
                //printf("k %d ", k);
                vtkm::Float64 f = aryData.GetPortalConstControl().Get(k)/(20*20*20);
                if(f==0) {
                  postProb[bincnt] = 0;
                  bincnt+=1;
                  k+=1;
                  continue;
                }
                k+=2;
                /*
                //if(f==0) continue;
                vtkm::worklet::GMM<4, 3, vtkm::Float64> gmm;
                //set gmm
                //printf("k = %d ", k);
                gmm.setGMM(aryData, k);
                //printf("%d\n", k);*/

                //std::cout << OrIdx << " f " << f << std::endl;
                int gmmid = (int)aryData.GetPortalConstControl().Get(k);
                k++;
                vtkm::Float64 p = em.gmmsHandle.GetPortalConstControl().Get(gmmid).getProbability(xyz_);//gmm.getProbability(xyz_);
                // std::cout<<"gmm "<<p<<std::endl;
                vtkm::Float32 Rprob = f*p;
                //std::cout << "Rp " << Rprob << std::endl;
                postProb[bincnt] = Rprob;
                postProbSum += postProb[bincnt];
                bincnt+=1;
            }
            // printf("blk %d\n", bincnt);
            //// Normalize /////
            for (int bin = 0; bin < numberOfBins; ++bin)
            {
              postProb[bin] = postProb[bin] / postProbSum;
            }
            
            vtkm::Float32 rUniform = dr(rng);
            // printf("random %f\n", rUniform);

            int k;
            vtkm::Float32 sum = 0;
            for (k = 0; k < numberOfBins; k++) {
              sum += postProb[k];
              if (sum > rUniform){
                break;
              }
            }
            if (k == numberOfBins) k--;
            
            vtkm::Float64 lo = fieldMinValue + (static_cast<vtkm::Float64>(k) * delta);
            vtkm::Float64 hi = lo + delta;
            vtkm::Float64 average = (lo + hi) / 2;
            vtkm::Float32 sample = average;

            vtkm::Vec<vtkm::Float32, 4> RGBD{0};
            transfer_func(RGBD, sample);
            float w = (1-total_alpha)*RGBD[3];
            total_alpha+=w;
            R += (RGBD[0]*255)*w;
            G += (RGBD[1]*255)*w;//0;
            B += (RGBD[2]*255)*w;//0;
            if(total_alpha>=0.99) break;
            
            // std::cout<<"rgbd "<<total_alpha<<" "<<RGBD[0]*255<<" "<<RGBD[1]*255<<" "<<RGBD[2]*255<<std::endl;
            xx += static_cast<int>(0.001*dir[0]);
            yy += static_cast<int>(0.001*dir[1]);
            zz += static_cast<int>(0.001*dir[2]);
            xyz_ = vtkm::make_Vec(xx, yy, zz);

          }while(xx <= boxSize[0] && yy <= boxSize[1] && zz<=boxSize[2]);

          
        }
        else{
          switch ((int)arr.GetPortalConstControl().Get(i))
          {
            case 1:
              R = 255;
              G = 0;
              B = 0;
              break;
            case 2:
              R = 255;
              G = 255;
              B = 0;
              break;
            case 3:
              R = 255;
              G = 0;
              B = 255;
              break;
            case 4:
              R = 0;
              G = 255;
              B = 0;
              break;
            case 5:
              R = 0;
              G = 0;
              B = 255;
              break;
            case 6:
              R = 0;
              G = 255;
              B = 255;
              break;
            default:
              R = 127;
              G = 127;
              B = 127;
          }
        }

        

        auto col = Pixel{ static_cast<unsigned char>(R), static_cast<unsigned char>(G), static_cast<unsigned char>(B) };
        img.writePixel(x, y, col);
      }
    }
    char filename[] = "output.ppm";
    img.outputPPM(filename);
  }

  void transfer_func(vtkm::Vec<vtkm::Float32, 4> &RGBD, vtkm::Float32 value)
  {
    float transfer[3][5]={{-4815, 0.231373, 0.298039, 0.752941, 0.1}, {-1221, 0.865003, 0.865003, 0.865003, 0.65}, {2372, 0.705882, 0.0156863, 0.14902, 0.99}};
    //float transfer[3][5]={{-4815, 0.1, 0.1, 0.9, 0}, {-1221, 0.75, 0.75, 0.75, 0.5}, {2372, 0.9, 0.015, 0.15, 1}};

    if(value>=transfer[0][0] && value<=transfer[1][0])
    {
      RGBD[0] = ((value-transfer[0][0])*(transfer[1][1]-transfer[0][1]))/(transfer[1][0]-transfer[0][0]) + transfer[0][1];
      RGBD[1] = ((value-transfer[0][0])*(transfer[1][2]-transfer[0][2]))/(transfer[1][0]-transfer[0][0]) + transfer[0][2];
      RGBD[2] = ((value-transfer[0][0])*(transfer[1][3]-transfer[0][3]))/(transfer[1][0]-transfer[0][0]) + transfer[0][3];
      RGBD[3] = ((value-transfer[0][0])*(transfer[1][4]-transfer[0][4]))/(transfer[1][0]-transfer[0][0]) + transfer[0][4];
    }
    else if(value>=transfer[1][0] && value<=transfer[2][0])
    {
      RGBD[0] = ((value-transfer[1][0])*(transfer[2][1]-transfer[1][1]))/(transfer[2][0]-transfer[1][0]) + transfer[1][1];
      RGBD[1] = ((value-transfer[1][0])*(transfer[2][2]-transfer[1][2]))/(transfer[2][0]-transfer[1][0]) + transfer[1][2];
      RGBD[2] = ((value-transfer[1][0])*(transfer[2][3]-transfer[1][3]))/(transfer[2][0]-transfer[1][0]) + transfer[1][3];
      RGBD[3] = ((value-transfer[1][0])*(transfer[2][4]-transfer[1][4]))/(transfer[2][0]-transfer[1][0]) + transfer[1][4];
    }
  }

  template <typename FieldType, typename Storage>
  void PrintArray(vtkm::cont::ArrayHandle<FieldType, Storage> arr)
  {
    int i ;
    for (i = 0; i < arr.GetNumberOfValues(); i++)
    {
      std::cout <<  arr.GetPortalConstControl().Get(i) <<' ';//<<  arr.GetPortalConstControl().Get(i)[1] << " ";
    }
    std::cout << '\n' ;
  }

  template<typename FieldType, typename Storage>
  void PrintT(vtkm::cont::ArrayHandle<FieldType, Storage> arr)
  {
    int i;
    for (i = 0; i < arr.GetNumberOfValues(); i++)
    {
      if (arr.GetPortalConstControl().Get(i) == -1)
        std::cout << " ";
      else
        printf("%d", i%10);
        //printf("%.3f", arr.GetPortalConstControl().Get(i));
      if (i % 100 == 0)
        std::cout << "|\n|";
    }
    std::cout << '\n';
  }
  // Execute the histogram binning filter given data and number of bins, min,
  // max values.
  // Returns:
  // number of values in each bin
  //template <typename FieldType, typename Storage  >
  void Run(vtkm::Vec<vtkm::Float32, 3> boxSize,
           vtkm::Vec<vtkm::Id, 2> imageResolution,
           vtkm::Vec<vtkm::Float32, 3> rotation,
           vtkm::Float32 angle,
           vtkm::cont::ArrayHandle<vtkm::Float32>& pixelValue,
           vtkm::cont::ArrayHandle<vtkm::Id> &aryOffset,
           vtkm::cont::ArrayHandle<vtkm::Float64> &aryData)
  {
    srand(time(NULL));
    auto temp = vtkm::Max(boxSize[0], vtkm::Max(boxSize[1], boxSize[2]));
    vtkm::Vec<vtkm::Float32, 3> cameraLoc = vtkm::make_Vec(0, 0, -1.5 * temp);
    vtkm::Vec<vtkm::Float32, 3> U = vtkm::make_Vec(0, 1, 0);
    if (rotation[0] != 0)
    {
      vtkm::Matrix<vtkm ::Float32, 3, 3> xRotation;
      xRotation[0][0] = 1;
      xRotation[0][1] = 0;
      xRotation[0][2] = 0;
      xRotation[1][0] = 0;
      xRotation[1][1] = vtkm::Cos(rotation[0]);
      xRotation[1][2] = -vtkm::Sin(rotation[0]);
      xRotation[2][0] = 0;
      xRotation[2][1] = vtkm::Sin(rotation[0]);
      xRotation[2][2] = vtkm::Cos(rotation[0]);
      cameraLoc = vtkm::MatrixMultiply(xRotation, cameraLoc);
      U = vtkm::MatrixMultiply(xRotation, U);
    }
    if (rotation[1] != 0)
    {
      vtkm::Matrix<vtkm ::Float32, 3, 3> yRotation;
      yRotation[0][0] = vtkm::Cos(rotation[1]);
      yRotation[0][1] = 0;
      yRotation[0][2] = vtkm::Sin(rotation[1]);
      yRotation[1][0] = 0;
      yRotation[1][1] = 1;
      yRotation[1][2] = 0;
      yRotation[2][0] = -vtkm::Sin(rotation[1]);
      yRotation[2][1] = 0;
      yRotation[2][2] = vtkm::Cos(rotation[1]);
      cameraLoc = vtkm::MatrixMultiply(yRotation, cameraLoc);
      U = vtkm::MatrixMultiply(yRotation, U);
    }
    if (rotation[2] != 0)
    {
      vtkm::Matrix<vtkm ::Float32, 3, 3> zRotation;
      zRotation[0][0] = vtkm::Cos(rotation[2]);
      zRotation[0][1] = -vtkm::Sin(rotation[2]);
      zRotation[0][2] = 0;
      zRotation[1][0] = vtkm::Sin(rotation[2]);
      zRotation[1][1] = vtkm::Cos(rotation[2]);
      zRotation[1][2] = 0;
      zRotation[2][0] = 0;
      zRotation[2][1] = 0;
      zRotation[2][2] = 1;
      cameraLoc = vtkm::MatrixMultiply(zRotation, cameraLoc);
      U = vtkm::MatrixMultiply(zRotation, U);
    }
    vtkm::Vec<vtkm::Float32, 3> viewDir = -1 * cameraLoc;
    vtkm::Normalize(viewDir);
    vtkm::Vec<vtkm::Float32, 3> horizontal = vtkm::Cross(viewDir, U);
    vtkm::Normalize(horizontal);
    vtkm::Vec<vtkm::Float32, 3> vertical = vtkm::Cross(horizontal, viewDir);
    vtkm::Normalize(vertical);
    horizontal = horizontal  * vtkm::Tan(angle / 180 * vtkm::Pi());
    vertical = vertical * vtkm::Tan(angle / 180 * vtkm::Pi());
    vtkm::Vec<vtkm::Float32, 3> imageUpperLeftLoc = cameraLoc + viewDir + horizontal + vertical;
    horizontal = horizontal / imageResolution[0] * 2;
    vertical = vertical / imageResolution[1] * 2;
    
    vtkm::cont::ArrayHandleCounting<vtkm::Id> imageIndex(0, 1, imageResolution[0]*imageResolution[1]);
    vtkm::cont::ArrayHandle<vtkm::Float32> checkHit;
    checkHit.Allocate(imageResolution[0] * imageResolution[1]);
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3> > xyz;
    xyz.Allocate(imageResolution[0] * imageResolution[1]);

    Intersect intersectWorklet(boxSize, cameraLoc, imageUpperLeftLoc, imageResolution, horizontal,vertical);
    vtkm::worklet::DispatcherMapField<Intersect> intersectDispatcher(intersectWorklet);
    intersectDispatcher.Invoke(imageIndex, checkHit, xyz);
    //PrintArray(xyz);

    /*
    if(checkHit.Get(i)!= -1 ){
      dir = xyz.Get(i) - cameraLoc;
      vtkm::Normalize(Dir);
      j = 1;
      while( (j*dir+cameraLoc) still in the box)
        i.color += j*dir+cameraLoc.color
        j++
    }
    */
    Output(boxSize, cameraLoc, checkHit, imageResolution, xyz, aryData, aryOffset);

  }
};
}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_VolumeRendering_h
