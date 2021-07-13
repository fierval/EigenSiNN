#pragma once

#include "network_graph.hpp"
#include "onnx/loader.h"

namespace EigenSinn {

	template <typename Scalar, typename Actual, typename Loss, typename Device_>
	class PersistentNetworkBase : public NetworkBase<Scalar, Actual, Loss, Device_>
	{
	public:
		typedef std::shared_ptr<EigenModel> PtrEigenModel;

		PersistentNetworkBase() {

		}

		PtrEigenModel save(bool for_training = true) {

			// need to make sure we have topologically sorted vertices
			assert(forward_order.size() > 0);
			PtrEigenModel model = std::make_shared<EigenModel>(for_training);

			vertex_name names = boost::get(boost::vertex_name, graph);

			// we need to save the model based on the topological order
			// or else we won't be able to define edges in the ONNX graph
			for (auto& v : forward_order) {
			
				std::string v_name = names(v);

				auto input_names = collect_input_names(v);

				// input layer.
				// the "output" of the input layer is the input layer itself
				if (input_names.empty()) {
					if (layer_output.count(v_name) == 0) {
						layer_output.insert(std::make_pair(v_name, v_name));
					}
					continue;
				}

				// collect output names: these are input edges into the layer
				// which are named after the input tensors
				std::vector<std::string> tensor_input_names(input_names.size());

				std::transform(input_names.begin(), input_names.end(), tensor_input_names.begin(), [&](std::string& in_layer) {
					return layer_output[in_layer];
					});

				// we'll get back the name of the output edge for this tensor and use it to
				// keep building the graph
				std::string layer_output_name = graph[v].layer->add_onnx_node(*model, tensor_input_names);
				layer_output.insert(std::make_pair(v_name, layer_output_name));
			}

			return model;
		}

		void load(EigenModel& model) {

		}

	private:

		std::vector<std::string> collect_input_names(vertex_t& v) {

			std::vector<std::string> input_names;
			InEdgeIter in, in_end;
			vertex_name names = boost::get(boost::vertex_name, graph);

			for (boost::tie(in, in_end) = boost::in_edges(v, graph); in != in_end; in++) {

				vertex_t inp = boost::source(*in, graph);
				input_names.push_back(names[inp]);
			}

			return input_names;
		}

		// REVIEW: this assumes only one actual output from a layer
		// output names of layer tensors
		std::map<std::string, std::string> layer_output;
	};
}