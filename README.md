# Optimizing-Agent-Model-Control

The entire code is in master, which contains an upload document. It contains 2,3,4,5 documents.
2. for data-driven modeling, mainly to establish the relationship between the input: power and output temperature change of the model, due to data confidentiality, so only part of the code is provided, of course, with matlab's systemidentification toolkit can also be completed. There are many powerful functions in it.
3. mainly design fuzzy controller and incremental fuzzy controller, it can be seen that incremental fuzzy controller is improved on the basis of fuzzy controller. And unit step response is performed. In the improvement process, the quantization of inputs and outputs need to be modified to map them to the correct region.
4.Mainly, several optimization algorithms are introduced for the help of controller parameters, simple optimization algorithms can be done using optimization module of matlab. The comparison related to GA,PSO and QNPSO is shown in the code.Optimization after that is very helpful for the controller. The simulation based on environment variables is not available due to data confidentiality.
5. The main use of neural networks, according to the input: environmental factors and output: optimization parameter set to establish a relationship, then we can avoid training optimization in the actual use of time but directly use the optimization parameters, so that the effect is better, this includes the actual experiments and simulation experiments based on the environment, due to the confidentiality of the data can not be provided.
Thank you for your time.
