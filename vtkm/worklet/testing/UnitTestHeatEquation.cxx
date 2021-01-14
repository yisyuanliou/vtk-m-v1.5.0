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

void TestHeatEquation()
{
  int nIters = 100;

  // Create Heat init Table
  vtkm::cont::DataSet tempTable;
  {
    std::vector<vtkm::Id> x = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    vtkm::worklet::VtkmSQL::AddColumn(tempTable, "X", x);
    std::vector<vtkm::Id> y = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    vtkm::worklet::VtkmSQL::AddColumn(tempTable, "Y", y);
    std::vector<vtkm::Float32> temp = {0, 1, 20, 100, 3, 0, 3, 0, 0, 4, 0, 8};
    vtkm::worklet::VtkmSQL::AddColumn(tempTable, "Temp", temp);
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable( tempTable );
  }

  //Adjacent stencil  Table
  vtkm::cont::DataSet stencilTable;
  {
    std::vector<vtkm::Id> x = {0, -1, 0, 1, 0};
    vtkm::worklet::VtkmSQL::AddColumn(stencilTable, "StencilX", x);
    std::vector<vtkm::Id> y = {-1, 0, 0, 0, 1};
    vtkm::worklet::VtkmSQL::AddColumn(stencilTable, "StencilY", y);
    std::vector<vtkm::Float32> weight = {0.125, 0.125, 0.5, 0.125, 0.125};
    vtkm::worklet::VtkmSQL::AddColumn(stencilTable, "StencilW", weight);
    vtkm::worklet::VtkmSQL::PrintVtkmSqlTable( stencilTable );
  }

  for( int i=0; i<nIters; i++ ){
    
    vtkm::cont::DataSet adjacentTable;
    {
      //Select tempTalbe.X, tempTable.Y tempTable.Z stencilTable.StencilX stencilTable.StencilY stencilTable.StencilW
      //  From tempTable stencilTable
      //  CrossJoin
      vtkm::worklet::VtkmSQL vtkmSql(tempTable, stencilTable);
      vtkmSql.Select(0, "X", "X");
      vtkmSql.Select(0, "Y", "Y");
      vtkmSql.Select(0, "Temp", "Temp");
      vtkmSql.Select(1, "StencilX", "StencilX");
      vtkmSql.Select(1, "StencilY", "StencilY");
      vtkmSql.Select(1, "StencilW", "StencilW");
      vtkmSql.CrossJoin();
      adjacentTable = vtkmSql.Query( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
      //vtkm::worklet::VtkmSQL::PrintVtkmSqlTable( adjacentTable );
    }

    vtkm::cont::DataSet propogateTempTable;
    {
      //Select ADD(X,StencilX) as X, ADD(X,StencilY) as Y, Mul(Temp,StencilW) as Temp
      //  From adjacentTable
      vtkm::worklet::VtkmSQL vtkmSql(adjacentTable);
      vtkmSql.Select(0, "X", "ADD", "StencilX", "X");
      vtkmSql.Select(0, "Y", "ADD", "StencilY", "Y");
      vtkmSql.Select(0, "Temp", "MUL","StencilW", "Temp");
      propogateTempTable = vtkmSql.Query( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
      //vtkm::worklet::VtkmSQL::PrintVtkmSqlTable( propogateTempTable );
    }

    //Averge by coordinate
    {
      //Select X, Y, SUM(Temp)
      // From propogateTempTable
      // Where 0<=X<=3 AND 0<=Y<=2
      // GroupBy X, Y
      vtkm::worklet::VtkmSQL vtkmSql(propogateTempTable);
      vtkmSql.Select(0, "X", "X");
      vtkmSql.Select(0, "Y", "Y");
      vtkmSql.Select(0, "Temp", "SUM", "Temp");
      std::vector<vtkm::Range> rangeX = {vtkm::Range(0, 3)};
      vtkmSql.Where(0, "X", rangeX);
      std::vector<vtkm::Range> rangeY = {vtkm::Range(0, 2)};
      vtkmSql.Where(0, "Y", rangeY);
      vtkmSql.GroupBy(0, "X");
      vtkmSql.GroupBy(0, "Y");

      tempTable = vtkmSql.Query( VTKM_DEFAULT_DEVICE_ADAPTER_TAG() );
      std::cout << "Iteration: " << i << std::endl;
      vtkm::worklet::VtkmSQL::PrintVtkmSqlTable( tempTable );
    }
  }

} // TestFieldHistogram
}

int UnitTestHeatEquation(int, char* [])
{
  return vtkm::cont::testing::Testing::Run(TestHeatEquation);
}
