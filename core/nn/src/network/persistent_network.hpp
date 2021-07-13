#pragma once

#include "network_graph.hpp"
#include "onnx/loader.h"

namespace EigenSinn {

	template <typename Scalar, typename Actual, typename Loss, typename Device_>
	class PersistentNetworkBase : public NetworkBase<Scalar, Actual, Loss, Device_>
	{
	public:
		PersistentNetworkBase() {

		}
	};
}