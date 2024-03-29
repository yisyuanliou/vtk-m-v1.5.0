#============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2014 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
##  Copyright 2014 UT-Battelle, LLC.
##  Copyright 2014 Los Alamos National Security.
##
##  Under the terms of Contract DE-NA0003525 with NTESS,
##  the U.S. Government retains certain rights in this software.
##
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##============================================================================

set(unit_tests
  UnitTestCellAverage.cxx
  UnitTestCellDeepCopy.cxx
  UnitTestCellGradient.cxx
  UnitTestClipping.cxx
  UnitTestContourTreeUniform.cxx
  UnitTestExternalFaces.cxx
  UnitTestExtractGeometry.cxx
  UnitTestExtractPoints.cxx
  UnitTestExtractStructured.cxx
  UnitTestFieldHistogram.cxx
  UnitTestFieldStatistics.cxx
  UnitTestKdTreeBuildNNS.cxx
  UnitTestKeys.cxx
  UnitTestMagnitude.cxx
  #UnitTestMarchingCubes.cxx
  UnitTestMask.cxx
  UnitTestMaskPoints.cxx
  UnitTestParticleAdvection.cxx
  UnitTestPointElevation.cxx
  UnitTestPointGradient.cxx
  UnitTestRemoveUnusedPoints.cxx
  UnitTestScatterCounting.cxx
  UnitTestScatterPermutation.cxx
  UnitTestSplatKernels.cxx
  UnitTestStreamingSine.cxx
  UnitTestStreamLineUniformGrid.cxx
  UnitTestSurfaceNormals.cxx
  UnitTestTetrahedralize.cxx
  UnitTestThreshold.cxx
  UnitTestThresholdPoints.cxx
  UnitTestTriangulate.cxx
  UnitTestWholeCellSetIn.cxx
  UnitTestWorkletMapField.cxx
  UnitTestWorkletMapFieldExecArg.cxx
  UnitTestWorkletMapFieldWholeArray.cxx
  UnitTestWorkletMapTopologyExplicit.cxx
  UnitTestWorkletMapTopologyUniform.cxx
  UnitTestWorkletReduceByKey.cxx
  UnitTestVertexClustering.cxx
  UnitTestWaveletCompressor.cxx
  UnitTestVolumeRendering.cxx
  spatialGMM.cxx
  
  #UnitTestHistModelAndReconstruction.cxx
  #UnitTestGMMTraining.cxx
  #UnitTestGMMReconstruction.cxx
  #UnitTestHeatEquation.cxx
 )

vtkm_unit_tests(
  SOURCES ${unit_tests}
  LIBRARIES vtkm_source vtkm_worklet vtkm_filter
  ALL_BACKENDS
  )
if (TARGET vtkm::cuda)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
    set(problematic_cuda_srcs
      "${CMAKE_CURRENT_BINARY_DIR}/UnitTestParticleAdvection.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/UnitTestBoundingIntervalHierarchy.cu"
    )
    set_source_files_properties(${problematic_cuda_srcs} PROPERTIES COMPILE_FLAGS "-Xptxas --opt-level=0")
  endif()
endif()