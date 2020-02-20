package neuralnetwork.network;

//import java.util.ArrayList;
import java.lang.Math;
import neuralnetwork.math.*;
import neuralnetwork.exceptions.*;

//A basic multi-layered perceptron
public class NeuralNetwork {
	
	private float[] inputs;
	private float[][] hidden;
	private float[] outputs;
	
	private Matrix[] weights;
	private float[][] biases;
	
	//CONSTRUCTOR
	//WARNING
	//DO NOT CREATE A CLASS WHERE LAYERS ARE LESS THAN THREE OR layer_neurons.length != layers
	public NeuralNetwork(int layers, int[] layer_neurons){
		
		//EXCEPTIONS
		if(layers != layer_neurons.length) {
			String message = "The array length must be equal to the number of layers";
			throw new NetworkStructureException(message);
		}
		else if(layers<3) {
			String message = "Number of layers must be greater than 3";
			throw new NetworkStructureException(message);
		}

		//INPUT NEURONS
		inputs = new float[layer_neurons[0]];
		
		//HIDDEN NEURONS
		hidden = new float[layers-2][];
		for(int x = 0; x<hidden.length; x++) {
			hidden[x] = new float[layer_neurons[x+1]];
		}
		//OUTPUT NEURONS
		outputs = new float[layer_neurons[layer_neurons.length - 1]];
		
		
		//INITIALIZE WEIGHTS AND BIASES
		weights = new Matrix[layers-1];
		biases = new float[layers-1][];
		
		//weights and biases between input and first hidden layer
		weights[0] = new Matrix(hidden[0].length, inputs.length);
		biases[0] = new float[hidden[0].length];
		
		//weights between layers of hidden
		for(int x = 0; x<hidden.length - 1;x++) {
			weights[x+1] = new Matrix(hidden[x+1].length, hidden[x].length);
			biases[x+1] = new float[hidden[x+1].length];
		}

		//weight matrix between last hidden layer and output
		weights[weights.length-1] = new Matrix(outputs.length, hidden[hidden.length-1].length);
		biases[biases.length - 1] = new float[outputs.length];
	}
	
	//Returns the total number of layers the network has
	public int Layers() {
		return hidden.length + 2;
	}
	
	//Print specifications of the neural network
	public String toString() {
		String specs = "Total layers: " + Layers() + "\n"+
					   "Inputs: " + inputs.length + "\n";
		for(int x = 0; x<hidden.length; x++) {
			specs += "Hidden Layer " + (x+1) + ": " + hidden[x].length + "\n";
		}
		//Weight Count
		int weightCount = 0;
		for(int x = 0; x<weights.length;x++){
			weightCount += weights[x].Rows()*weights[x].Cols();
		}
		//Biases Count
		int biasesCount = 0;
		for(int x = 0; x<biases.length;x++) {
			biasesCount += biases[x].length;
		}
		specs+="Total Weights: "+weightCount+"\n";
		specs+="Total Biases: "+biasesCount+"\n";
		specs+="Outputs: " +outputs.length;
		return specs;
	}
	
	//Returns the answer of the neural network in a form of array
	public float[] FeedForward(float[] i_) {
		
		//Input to first hidden
		Matrix inputM = new Matrix(i_,true); // put in matrix object
		Matrix passed = Matrix.Mul(weights[0], inputM); //matrix multiply
		passed.Add(new Matrix(biases[0], true)); // add biases
		hidden[0] = Activation(passed).GetColumn(0);//pass to first hidden layer
		
		//pass the answers forward the hidden layers
		for(int x = 1; x<hidden.length;x++)	{
			inputM = new Matrix(hidden[x-1], true);//put previous layer in matrix object
			passed = Matrix.Mul(weights[x], inputM);//matrix multiply
			passed.Add(new Matrix(biases[x], true));//add biases
			hidden[x] = Activation(passed).GetColumn(0);//pass to next layer
		}
		
		//Pass the last hidden layer to get output
		inputM = new Matrix(hidden[hidden.length-1], true);//put last hidden layer to matrix object
		passed = Matrix.Mul(weights[weights.length-1], inputM);//matrix mulitply
		passed.Add(new Matrix(biases[biases.length - 1], true));// add bias
		outputs = Activation(passed).GetColumn(0);//pass to output layer
			
		return outputs;
	}
	
	public void Backpropagate(float[] target){
		return;
	}

	//Sigmoid function
	float Activation(float input){
		return (float)(1/(Math.pow(2.718f, -input) + 1));
	}
	
	Matrix Activation(Matrix m) {
		Matrix n = new Matrix(m.Rows(), m.Cols());
		for(int x = 0; x<m.Rows(); x++)	{
			for(int y = 0;y<m.Cols(); y++) {
				n.SetCell(Activation(m.GetCell(x, y)), x, y); ;
			}
		}
		return n;
	}
}