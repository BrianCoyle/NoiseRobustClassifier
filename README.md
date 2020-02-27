# NoiseRobustClassifier
Code to implement numerical results in "Robust Data Encodings for Quantum Classifiers" by Ryan LaRose & Brian Coyle
arXiv:

Running these functions will generate the figures seen in the above paper.


``python3 visualise_datasets.py``

- Generates Figure 4 for each 2d Dataset studied.


``python3 visualise_decision_boundaries.py``

- Generates Figure 6 for each encoding strategy.

``python3 plots.py``

- Generates Figure 8 and Figure 13 for Pauli and Measurement Noise in the single qubit classifier.

``python3 encoding_learning_algorithm.py``

- Generates Figure 10 in the single qubit classifier, for each dataset and encoding strategy.

``python3 encoding_learning_algorithm.py``

- Generates Figure 10 in the single qubit classifier, for each dataset and encoding strategy.

``python3 decision_boundary_vertical.py`` setting 'compare' = True

- Generates Figure 11 in the single qubit classifier, for the 'vertical' dataset. Compares all encoding parameters
in the noisy and noiseless case.

``python3 decision_boundary_vertical.py`` setting:'analytic'=True, 'ideal'=True, 'noise'=True

- Generates Figure 7 in the single qubit classifier, for the 'vertical' dataset using Denseangle Encoding. Tests analytic 
condition deciding misclassification. Set 'encoding' = 'wavefunction_param' for results shown in Figure 14 (Appendix)


``python3 fidelity_analysis_two_qubit.py``

 
-Generates Fig 12 for the two qubit classifier on the Iris dataset, for each type of noise.
*NOTE:* This approach does not use the Rigetti simulator, instead simulates the density matrices directly.

