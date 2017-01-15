'''
	A Neural Network
	November 12, 2016
	Johnathan P. Irvin
'''

# Uses random for random generation of weight values
# Math because we're smart
import random, math

def Sigmoid(X, Derivative=False):
	if(Derivative == True):
		return X * (1 - X)
	else:
		return 1 / (1 + math.exp(-X))
	
class Connection:
	Total = 0
	
	def __init__(self, input_neuron, output_neuron):
		# Increase the static variable for total connections by 1
		Connection.Total += 1
		
		# The last connection in the current index of the new neuron
		# Indexes are unique keys
		self.Index = Connection.Total
		
		# This is the weight, we make it between 0 and 1
		self.Weight = random.randrange(0, 2)
		
		# This is the change in weight
		self.Delta_Weight = 0
		
		# Sets the instance's output neuron to the references provided
		self.Output_Neuron = output_neuron
		
		# Prints out that this connection was created
		print("Connection #" + str(self.Index) + " has been created with output neuron #" + str(self.Output_Neuron.Index))
	

class Neuron:
	Total = 0
	Training_Rate = 2
	Momentum = 2
	
	def __init__(self):
		Neuron.Total += 1
		self.Index = Neuron.Total
		
		# Network Starts with Zero Connections
		# Its an isolated spherical body of empowerment
		self.Total_Connections = 0
		self.Connections = []
		
		# What is currently stored inside this neuron
		self.Value = random.randrange(0, 50)
		
		self.Gradient = 0
		
		# Prints out that this neuron was created
		print("Neuron #" + str(self.Index) + " has been created")
		
	# The Neuron becomes the output neuron
	# The connection directs to the input, which is usually in the layer before
	def Add_Connection(self, Neuron_Connection):
		self.Connections.append(Neuron_Connection)
		self.Total_Connections += 1	
		
	# Function will connect the entire said layer to the current neuron
	def Connect_Layer(self, Layer_To_Connect):
		for Input_Neuron in Layer_To_Connect.Neurons:
			New_Connection = Connection(Input_Neuron, self)
			self.Add_Connection(New_Connection)
			
	def Calculate_Output_Gradient(self, Target_Value):	
		Delta = Target_Value - self.Value
		self.Gradient = Delta * Sigmoid(self.Value, True)
		
	def Calculate_Hidden_Gradient(self, Next_Layer):
		# Lets find out how much this neuron contributed to the error
		# So lets sum it up, we'll use this as our holder variable
		Sum_Of_Contributions_To_Error = 0.00
		
		# For each neuron in the next layer
		for Current_Neuron in Next_Layer.Neurons:
			# For each connection that neuron has
			for Current_Connection in Current_Neuron.Connections:
				# Add it to the sum of contributions
				Sum_Of_Contributions_To_Error += (Current_Neuron.Value * Current_Neuron.Gradient)
			
		# Contributions to error sum multipled the sigmoid of the output
		self.Gradient =  Sum_Of_Contributions_To_Error * Sigmoid(self.Value, True)
		
	def Update_Input_Weights(self, Previous_Layer):
		# For all neurons in the previous layer
		for Current_Neuron in Previous_Layer.Neurons:
			# For all connections the neuron has
			for Current_Connection in Current_Neuron.Connections:				
				if str(Current_Connection.Output_Neuron) == str(self):
					print(True)
					
				if Current_Connection.Output_Neuron == self:
					# Set a variable to store the delta weight before it is changed.
					Old_Delta_Weight = Current_Connection.Delta_Weight
					
					# Individual Input, Magnified By The Gradient and Train Rate will cause you to get the new delta weight
					New_Delta_Weight = (Neuron.Training_Rate * Current_Neuron.Value * self.Gradient) + (Neuron.Momentum * Old_Delta_Weight)
					
					# The Delta Weight is set to the secondary variable delta weight, while the actual weight is increased by the delta weight
					Current_Connection.Delta_Weight = New_Delta_Weight
					Current_Connection.Weight += New_Delta_Weight
	
class Layer:
	Total = 0
	def __init__(self, Neuron_Count):
		self.Neuron_Count = 0 
		self.Neurons = []
		
		Layer.Total += 1
		
		self.Index = Layer.Total
		
		for X in range(Neuron_Count):
			New_Neuron = Neuron()
			self.Add_Neuron(New_Neuron)
			
		# Prints out that this layer was created
		print("Layer #" + str(self.Index) + " has been created with " + str(self.Neuron_Count) + " total neurons")
		
	def Add_Neuron(self, Neuron):
		self.Neuron_Count += 1
		self.Neurons.append(Neuron)
		
	def Connect_Layer(self, Layer_To_Connect):
		# For all neurons in this current layer
		for Current_Neuron in self.Neurons:
			# Connect the entire input layer to that output neuron
			Current_Neuron.Connect_Layer(Layer_To_Connect)
		
class Network:
	Total = 0
	
	def __init__(self, Layer_Tuple):
		# Unique Network Identifier
		Network.Total += 1
		self.Index = Network.Total

		# Network Starts with Zero Layers
		self.Total_Layers = 0
		self.Layers = []
		
		# Create a layer for each number in the Tuple
		for Number in Layer_Tuple:
			# Create a new layer instance which the number of neurons in the Tuple
			New_Layer = Layer(Number)
			if(self.Total_Layers != 0):
				# If not the input layer, connect all neurons to the previous layer
				# The previous layer will be -1 of the total layers as where the layer we're going to add doesn't technically exist in the network yet.
				# Computer numbering is one behind because it starts with zero, and total layers is a literal count
				New_Layer.Connect_Layer(self.Layers[self.Total_Layers - 1])
				
			# Add this layer to the current network total layers
			self.Add_Layer(New_Layer)
		
		# Prints out that this network was created
		print("Network #" + str(self.Index) + " has been created with " + str(self.Total_Layers) + " total layers")

	def Add_Layer(self, layer):
		self.Total_Layers += 1
		self.Layers.append(layer)
		
	def Feed_Forward(self, Input_Values):
		# The Input_Layer is always the first layer of the network
		Input_Layer = self.Layers[0]
		# For each neuron in the input layer, set it to the current input value
		for X in range(len(Input_Layer.Neurons)):
			# Sets the current value to the corresponding tuple value
			Input_Layer.Neurons[X].Value = Input_Values[X]
			
			# Create a reference to the current neorn
			Current_Neuron = Input_Layer.Neurons[X]
			
			# Prints the new value
			print("Input Neuron #" + str(Current_Neuron.Index) + " has a new value of " + str(Current_Neuron.Value))
			
		for Layer_Counter in range(self.Total_Layers):
			# For all output neurons in the next layer
			for Output_Neuron in self.Layers[Layer_Counter].Neurons:
				# The total of all inputs, multiplied by their weights
				Sum_Of_Inputs = 0.00
				
				# For all input neurons the current layer has
				for Input_Neuron in self.Layers[Layer_Counter - 1].Neurons:
					# For all connections selected input neuron has
					for Current_Connection in Input_Neuron.Connections:
						if Current_Connection.Output_Neuron == Input_Neuron:
							# Multiply the output times the weight and add it to the overall total
							Sum_Of_Inputs += Current_Connection.Output_Neuron.Value * Current_Connection.Weight

				# Sets the neuron to its new value using the activation function
				Output_Neuron.Value = Sigmoid(Sum_Of_Inputs)
				
				# Print out the new value of the Neuron
				print("Neuron #" + str(Output_Neuron.Index) + " has a new value of " + str(Output_Neuron.Value))
			

	def Backwards_Propagation(self, Target_Values):
		# This will be used to tell us what our Error Rate is
		self.Error_Rate = 0.00
		
		# The output layer is always the last layer of the network
		Output_Layer = self.Layers[self.Total_Layers - 1]
		
		# Calculate The Error Rate
		# For all neurons in the output layer, using the counter in the range.
		# Root mean square error
		for X in range(Output_Layer.Neuron_Count):
			Delta = Target_Values[X] - Output_Layer.Neurons[X].Value
			self.Error_Rate = Delta * Delta
			
		# Divide by the total neurons
		self.Error_Rate /= Output_Layer.Neuron_Count
		# Sqrt the number
		self.Error_Rate = math.sqrt(self.Error_Rate)
		
		print("Error Rate: " + str(self.Error_Rate))
		
		# Calculate Output Layer Gradients
		# Count the entire amount of values
		for X in range(len(Target_Values)):
			# For each value find the corresponding neuron, and target value
			# Calculate the output gradient using the instance function
			Output_Layer.Neurons[X].Calculate_Output_Gradient(Target_Values[X])
		
		# Calculate Hidden Layer Gradients
		# Which is all layers that aren't the first or the last layer
		for X in range(self.Total_Layers - 2, 0, -1):
			# The hidden layer is always going to be this X
			Hidden_Layer = self.Layers[X]
			# The next layer may not be a hidden layer
			Next_Layer = self.Layers[X + 1]
			
			# For each neuron in the hidden layer
			for Current_Neuron in Hidden_Layer.Neurons:
				# Calculate the gradient using the next layer
				Current_Neuron.Calculate_Hidden_Gradient(Next_Layer)
			
		# Update All Layers
		# For the count of total layers in the internal layer count
		for X in range(self.Total_Layers):
			# As long as not first layer
			if(X > 0):
				# for all neurons in the current layer
				for Current_Neuron in self.Layers[X].Neurons:
					# Updates all weights based on previous layer
					Current_Neuron.Update_Input_Weights(self.Layers[X - 1])
			
	def Get_Results(self):
		# For each neuron in the output layer
		for Current_Neuron in self.Layers[self.Total_Layers - 1].Neurons:
			print("The value output of output neuron #" + str(Current_Neuron.Index) + " is " + str(Current_Neuron.Value))
		
def Main():
	Neural_Net = Network([2, 3, 1])
	Epoch = 1
	
	while(Epoch < 2000):
		print (str(Epoch))
		Input_Values = [0, 1]
		Neural_Net.Feed_Forward(Input_Values)
		Target_Values = [1]
		Neural_Net.Backwards_Propagation(Target_Values)
		
		Neural_Net.Get_Results()
		Epoch += 1
		
		print (str(Epoch))
		Input_Values = [1, 1]
		Neural_Net.Feed_Forward(Input_Values)
		Target_Values = [0]
		Neural_Net.Backwards_Propagation(Target_Values)
		
		Neural_Net.Get_Results()
		Epoch += 1
		
		print (str(Epoch))
		Input_Values = [1, 0]
		Neural_Net.Feed_Forward(Input_Values)
		Target_Values = [1]
		Neural_Net.Backwards_Propagation(Target_Values)
		
		Neural_Net.Get_Results()
		Epoch += 1
		
		print (str(Epoch))
		Input_Values = [0, 0]
		Neural_Net.Feed_Forward(Input_Values)
		Target_Values = [0]
		Neural_Net.Backwards_Propagation(Target_Values)

		Neural_Net.Get_Results()
		Epoch += 1
	
Main()