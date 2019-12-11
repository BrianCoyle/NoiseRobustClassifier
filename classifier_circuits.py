import numpy as np
from pyquil.gates import *
from noise_models import add_noisy_gate_to_circ
from noise_models import *
from pyquil.noise import add_decoherence_noise

class NoisyCircuits():

    def __init__(self, circuit, params, qubits, noise=None, noise_params=None):
        self.circuit = circuit
        self.noise_model = noise
        self.params = params
        self.qubits = qubits
        
        if noise == None:
            self._ideal_circuit()
        else:
            self.noise_params = noise_params

            if   self.noise_model.lower() == "depolarizing_before_meas":    self._dephased_before_meas_circuit()
            elif self.noise_model.lower() == "pauli_before_measurement":    self._pauli_before_meas_circuit()
            elif self.noise_model.lower() == "decoherence":                 self._ideal_circuit() # Decoherence noise model must be added to compiled gates
            elif self.noise_model.lower() == "measurement":                 self._measurement_noise()
            elif self.noise_model.lower() == "amp_damp_before_measurement": self._amp_damp_before_meas_circuit()

        

    def _ideal_circuit(self):
        """
        Writes the ideal classification ansatz circuit.
        For now, just general single qubit unitary decomposed into RxRzRx
        with three parameters
        """
        
        self.circuit += RZ(2*self.params.values[0][0], self.qubits[0])
        self.circuit += RY(2*self.params.values[0][1], self.qubits[0])
        self.circuit += RZ(2*self.params.values[0][2], self.qubits[0])

        return self.circuit  


    def _dephased_before_meas_circuit(self):
        """
        Writes ideal classification circuit with dephasing noise added before measurement
        """
        p_dephase = self.noise_params 
        self._ideal_circuit()
        self.circuit += I(self.qubits[0])
        noisy_i = dephased_i_gate(p_dephase)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_i, [self.qubits[0]], 'iden')

        return self.circuit  


    def _pauli_before_meas_circuit(self):
        """
        Writes ideal classification circuit with pauli noise added before measurement
        """
        [p_I, p_x, p_y, p_z] = self.noise_params
        
        self._ideal_circuit()

        self.circuit += I(self.qubits[0])
        noisy_iden = pauli_noise_i_gate(p_I, p_x, p_y, p_z)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_iden, [self.qubits[0]], 'iden')

        return self.circuit  

    def _amp_damp_before_meas_circuit(self):
        """
        Writes ideal classification circuit with amplitude damping noise added before measurement
        """
        p_damp = self.noise_params 
        self._ideal_circuit()
        self.circuit += I(self.qubits[0])
        noisy_iden = amp_damp_i_gate(p_damp)

        self.circuit = add_noisy_gate_to_circ(self.circuit, noisy_iden, [self.qubits[0]], 'iden')

        return self.circuit 

    def _measurement_noise(self):
        """
        Adds asymmetric measurement noise to the circuit
        """
        [p00, p11] = self.noise_params # Noise values for readout error. Assume same probilities for every qubit
        
        self._ideal_circuit()

        for q in self.qubits:
            self.circuit.define_noisy_readout(q, p00, p11)

        return self.circuit 

    def decoherence_noise_model(self, *args):

        """
        Adds the Rigetti decoherence noise model to the circuit

        The default noise parameters
        - T1 = 30 us                T1           = 30e-6
        - T2 = 30 us                T2           = 30e-6
        - 1q gate time = 50 ns      gate_time_1q = 50e-9
        - 2q gate time = 150 ns     gate_time_2q = 150e-09
                                    ro_fidelity  = 0.95
        """
        [T1, T2, gate_time_1q, gate_time_2q, ro_fidelity] = args[0] 

        self.circuit = add_decoherence_noise(self.circuit, T1=T1, T2=T2, gate_time_1q=gate_time_1q, gate_time_2q=gate_time_2q, ro_fidelity=ro_fidelity)

        return self.circuit
