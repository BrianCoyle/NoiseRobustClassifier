import numpy as np
from pyquil.gates import *
from noise_models import add_noisy_gate_to_circ
from noise_models import *
from pyquil.noise import add_decoherence_noise

class NoisyCircuits():

    def __init__(self, circuit, params, device_qubits, noise=None, noise_params=None, qc=None, num_shots=100):
        self.circuit = circuit
        self.noise_model = noise
        self.params = params
        self.device_qubits = device_qubits
        self.n_layers = len(list(self.params.values.items())[0])

        if noise == None:
            self._ideal_circuit()
        else:
            self.noise_params = noise_params
            
            if   self.noise_model.lower() == "depolarizing_before_meas":    self._dephased_before_meas_circuit()
            elif self.noise_model.lower() == "pauli_before_measurement":    self._pauli_before_meas_circuit()
            elif self.noise_model.lower() == "decoherence_symmetric_ro":    
                self._ideal_circuit()

                ro = circuit.declare('ro', 'BIT', 1)
                self.circuit += MEASURE(self.device_qubits[0], ro[0])

                if qc is not None:
                    self.nativize(qc)
                self.decoherence_noise_model()
                
            elif self.noise_model.lower() == "measurement":                 self._measurement_noise()
            elif self.noise_model.lower() == "amp_damp_before_measurement": self._amp_damp_before_meas_circuit()
            else: raise ValueError("Error model not defined")
        

    def _ideal_circuit(self):
        """
        Writes the ideal classification ansatz circuit.
        For now, just general single qubit unitary decomposed into RxRzRx
        with three parameters
        """
        #######################################################################
        # TTN Classifier
        # self.circuit += RY(self.params.values[0][0], self.qubits[0])
        # self.circuit += RY(self.params.values[1][0], self.qubits[1])
        # self.circuit += RY(self.params.values[2][0], self.device_qubits[2])
        # self.circuit += RY(self.params.values[3][0], self.device_qubits[3])
        # self.circuit += CNOT(self.device_qubits[0], self.device_qubits[1])
        # self.circuit += CNOT(self.device_qubits[3], self.device_qubits[2])
        # self.circuit += RY(self.params.values[1][1], self.device_qubits[1])
        # self.circuit += RY(self.params.values[2][1], self.device_qubits[2])
        # self.circuit += CNOT(self.device_qubits[1], self.device_qubits[2])
        # self.circuit += RY(self.params.values[2][2], self.device_qubits[2])
        #TTN Ideal Params:  params.values = [0.12629255,  2.25997405,  0.67599152,  2.59226323,  0.31532559,  0.42584651, -1.07522358]
        # NEED TO CHANGE MEASUREMENT QUBIT TO QUBIT *2*
        #######################################################################
        first_qubit_index = self.device_qubits[0]
        if len(self.device_qubits) == 1:

            self.circuit += RZ(2*self.params.values[first_qubit_index][0][0], first_qubit_index)
            self.circuit += RY(2*self.params.values[first_qubit_index][0][1], first_qubit_index)
            self.circuit += RZ(2*self.params.values[first_qubit_index][0][2], first_qubit_index)
        elif len(self.device_qubits) == 2:
            # Arbitrary Two qubit unitary decompostions (up to a global phase) from 
            ## G. Vidal and C.M. Dawson Phys. Rev. A 69, 010301(R) 
            ## M Blaauboer and R L de Visser 2008 J. Phys. A: Math. Theor. 41 395307
            
            # Layer 1
            for ii, qubit in enumerate(self.device_qubits):
                # First three parameters for each qubit give general single qubit unitary. Will be overparameterized.
                self.circuit += RZ(2*self.params.values[ii][0][0], qubit)
                self.circuit += RY(2*self.params.values[ii][0][1], qubit)
                self.circuit += RZ(2*self.params.values[ii][0][2], qubit)

            # Layer 2    
            self.circuit += CNOT(self.device_qubits[0], self.device_qubits[1])
            self.circuit += H(self.device_qubits[0])
            self.circuit += RX(2*self.params.values[0][1][0] + np.pi, self.device_qubits[0])
            self.circuit += RZ(2*self.params.values[1][1][0], self.device_qubits[1])

            # # Layer 3
            self.circuit += CNOT(self.device_qubits[0], self.device_qubits[1])
            # self.circuit += H(self.device_qubits[0])
            # self.circuit += RZ(-2*self.params.values[1][2][0], self.device_qubits[1])
            
            # # Layer 4
            # self.circuit += CNOT(self.device_qubits[0], self.device_qubits[1])
            
            # self.circuit += RZ(np.pi/2, self.device_qubits[0])

            # self.circuit += RZ(2*self.params.values[0][3][0],  self.device_qubits[0])
            # self.circuit += RY(2*self.params.values[0][3][1],  self.device_qubits[0])
            # self.circuit += RZ(2*self.params.values[0][3][2],  self.device_qubits[0])

            # Layered circuit
            # for layer in range(self.n_layers):  
            #     # parameters should be an array of size n_device_qubits x n_layers x 3
            #     for ii, qubit in enumerate(self.device_qubits):
            #         self.circuit += RZ(2*self.params.values[ii][layer][0], qubit)
            #         self.circuit += RY(2*self.params.values[ii][layer][1], qubit)
            #         self.circuit += RZ(2*self.params.values[ii][layer][2], qubit)
            #     self.circuit += CZ(self.device_qubits[0], self.device_qubits[1]) # Unparameterized entanglement gate

        return self.circuit  

    def nativize(self, qc):
        """
        Take ideal circuit and compile to hardware of qc. Produce a set of native gates
        """
        self.circuit = qc.compiler.quil_to_native_quil(self.circuit)
        return self.circuit


    def _dephased_before_meas_circuit(self):
        """
        Writes ideal classification circuit with dephasing noise added before measurement
        """
        p_dephase = self.noise_params 
        self._ideal_circuit()
        self.circuit += I(self.device_qubits[0])
        noisy_i = dephased_i_gate(p_dephase)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_i, [self.device_qubits[0]], 'iden')

        return self.circuit  


    def _pauli_before_meas_circuit(self):
        """
        Writes ideal classification circuit with pauli noise added before measurement
        """
        [p_I, p_x, p_y, p_z] = self.noise_params
        
        self._ideal_circuit()

        self.circuit += I(self.device_qubits[0])
        noisy_iden = pauli_noise_i_gate(p_I, p_x, p_y, p_z)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_iden, [self.device_qubits[0]], 'iden')

        return self.circuit  

    def _amp_damp_before_meas_circuit(self):
        """
        Writes ideal classification circuit with amplitude damping noise added before measurement
        """
        p_damp = self.noise_params 
        self._ideal_circuit()
        self.circuit += I(self.device_qubits[0])
        noisy_iden = amp_damp_i_gate(p_damp)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_iden, [self.device_qubits[0]], 'iden')

        return self.circuit 

    def _measurement_noise(self):
        """
        Adds asymmetric measurement noise to the circuit
        """
        [p00, p11] = self.noise_params # Noise values for readout error. Assume same probilities for every qubit
        
        self._ideal_circuit()

        for q in self.device_qubits:
            self.circuit.define_noisy_readout(q, p00, p11)

        return self.circuit 

    def decoherence_noise_model(self):

        """
        Adds the Rigetti decoherence noise model to the circuit

        The default noise parameters
        - T1 = 30 us                T1           = 30e-6
        - T2 = 30 us                T2           = 30e-6
        - 1q gate time = 50 ns      gate_time_1q = 50e-9
        - 2q gate time = 150 ns     gate_time_2q = 150e-09
                                    ro_fidelity  = 0.95
        """

        [T1, T2, gate_time_1q, gate_time_2q, ro_fidelity] = self.noise_params

        self.circuit = add_decoherence_noise(self.circuit, T1=T1, T2=T2, gate_time_1q=gate_time_1q, gate_time_2q=gate_time_2q, ro_fidelity=ro_fidelity)

        return self.circuit
