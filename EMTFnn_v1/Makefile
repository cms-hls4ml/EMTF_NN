.PHONY: clean

EMTFnn_v1.so: emtfnn.o emtfnn_model_v1.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm EMTFnn_v1.so
