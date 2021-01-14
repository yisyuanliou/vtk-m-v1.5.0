#ifndef vtk_m_worklet_AplusB_h
#define vtk_m_worklet_AplusB_h

#include<vtkm/cont/ArrayHandle.h>
#include<vtkm/cont/ArrayHandleZip.h>

#include<vtkm/worklet/WorkletMapField.h>
#include<vtkm/worklet/DispatcherMapField.h>

namespace vtkm
{
	namespace worklet
	{
		class AplusB
		{
		public:
			class plus : public vtkm::worklet::WorkletMapField
			{
			public:
				using ControlSignature = void(FieldIn inputFirst, FieldIn inputSecond, FieldOut output);
				using ExecutionSignature = void(_1, _2, _3);
				//using InputDomain = _1, _2;

				template <typename FirstType, typename SecondType, typename resultType>
				VTKM_EXEC void operator()(FirstType first, SecondType second, resultType& result) const
				{
					result = first + second;
				}
			};

			//template <typename T>
			void Run(vtkm::cont::ArrayHandle<vtkm::Float32>& Array1,
				vtkm::cont::ArrayHandle<vtkm::Float32>& Array2,
				vtkm::cont::ArrayHandle<vtkm::Float32>& Array3)
			{
				vtkm::worklet::DispatcherMapField<plus> dispatcher;
				dispatcher.Invoke(Array1, Array2, Array3);
			}
		};
	}
}// namespace vtkm::worklet

#endif 