package neuralnetwork.network;

//import java.util.ArrayList;
import java.lang.Math;
import neuralnetwork.math.*;
import neuralnetwork.exceptions.*;

//A basic multi-layered perceptron
public class NeuralNetwork {
	
	private float lr; //Learning Rate
	private float[] inputs;
	private float[][] hidden;
	private float[] outputs;
	
	private Matrix[] weights;
	private float[][] biases;
	
	//CONSTRUCTOR
	//WARNING
	//DO NOT CREATE A CLASS WHERE LAYERS ARE LESS THAN THREE OR layer_neurons.length != layers
	public NeuralNetwork(int layers, int[] layer_neurons, float learning_rate){
		
		//EXCEPTIONS
		if(layers != layer_neurons.length) {
			String message = "The array length must be equal to the number of layers";
			throw new NetworkStructureException(message);
		}
		else if(layers<3) {
			String message = "Number of layers must be greater than 3";
			throw new NetworkStructureException(message);
		}

		lr = learning_rate;
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
		
		//randomize weights & biases
		for(int x =0; x<weights.length;x++) {
			weights[x].Randomize();
		}
		for(int x =0; x<biases.length; x++) {
			for(int y =0; y<biases[x].length;y++) {
				biases[x][y] = (float)Math.random();
			}
		}
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
		inputs = i_;
		Matrix inputM = new Matrix(inputs,true); // put in matrix object
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
	
	public void Backpropagate(float[] i_, float[] target){
		float[][] errors = new float[weights.length][];
		float[] guess = FeedForward(i_);
		//output to last hidden layer;
		errors[errors.length - 1] = new float[outputs.length];//Instantiate last layer of errors
		for(int x = 0; x<errors[errors.length-1].length; x++) {
			errors[errors.length-1][x] = 2*(guess[x] - target[x]); //put derivatives of the cost to last layer of errors
		}
		
		//CALCULATE D COST FUNCTION WITH RESPECT TO PREV ACTIVATION (OUTPUT TO HIDDEN)
		//errors * prime sigma * weight //formula for d cost function with respect to the prev activation prime sigma is just the value of the previous neuron
		errors[errors.length - 2] = new float[hidden[hidden.length -1].length];//Instantiate to the size of last hidden layer
		Matrix a1 = new Matrix(outputs);//Sigmoid
		Matrix current_e = new Matrix(outputs);
		//Transpose (sigmoid -1) to (-sigmoid + 1)
		current_e.ScalarMul(-1);
		current_e.ScalarAdd(1); //1-sigmoid
		
		current_e.HadamardProduct(a1);//sigmoid * (1 - sigmoid)
		Matrix e = new Matrix(errors[errors.length -1]);//Put current error to matrix
		current_e.HadamardProduct(e);//error * sigmoid * (1 - sigmoid)
		
		
		//D COST WITH RESPECT TO PREV ACTIVATION
		Matrix prev_e = Matrix.Mul(current_e, weights[weights.length -1]);//error * sigmoid * (1 - sigmoid) * weights
		errors[errors.length-2] = prev_e.GetRow(0);//put to previous layer errors
		current_e.ScalarMul(lr);//error * sigmoid * (1 - sigmoid)*learningRate;
		Matrix b = new Matrix(biases[biases.length - 1]);

		//Tune weight and bias here using current_e;
		//D COST WITH RESPECT TO BIAS
		b.Sub(current_e);
		biases[biases.length-1] = b.GetRow(0);
		
		//D COST WITH RESPECT TO WEIGHTS
		//Transpose matrix
		Matrix current_e_t = new Matrix(current_e.GetRow(0), true);
		Matrix prevActivation = new Matrix(hidden[hidden.length-1]);
		Matrix adjustments = Matrix.Mul(current_e_t, prevActivation);
		weights[weights.length -1].Sub(adjustments);
		

		for(int x = errors.length - 2; x>=1; x--) {
			errors[x-1] = new float[hidden[x].length];
			a1 = new Matrix(hidden[x]);
			current_e = new Matrix(hidden[x]);
			
			current_e.ScalarMul(-1);
			current_e.ScalarAdd(1);
			current_e.HadamardProduct(a1);
			e = new Matrix(errors[x]);
			current_e.HadamardProduct(e);
			
			
			prev_e = Matrix.Mul(current_e, weights[x]);//error * sigmoid * (1 - sigmoid) * weights
			errors[x-1] = prev_e.GetRow(0);
			
			//Tune weight and bias here using current_e;
			b = new Matrix(biases[x]);
			current_e.ScalarMul(lr);
			b.Sub(current_e);
			
			current_e_t = new Matrix(current_e.GetRow(0), true);
			prevActivation = new Matrix(hidden[x-1]);
			adjustments = Matrix.Mul(current_e_t, prevActivation);
			weights[x].Sub(adjustments);
			
		}
		
		//adjust input weights and bias here accourding to first index of errors
		a1 = new Matrix(hidden[0]);
		current_e = new Matrix(hidden[0]);
		
		current_e.ScalarMul(-1);
		current_e.ScalarAdd(1);
		current_e.HadamardProduct(a1);
		
		e = new Matrix(errors[0]);
		current_e.HadamardProduct(e);
		
		//Tune biases
		b = new Matrix(biases[0]);
		current_e.ScalarMul(lr);
		b.Sub(current_e);
		
		//Tune weights
		current_e_t = new Matrix(current_e.GetRow(0),true);
		prevActivation = new Matrix(inputs);
		adjustments= Matrix.Mul(current_e_t, prevActivation);
		weights[0].Sub(adjustments);
		
	}
	
	//Get the summation of an array
	float Summation(float[] col){
		float sum = 0;
		for(int x = 0; x<col.length;x++) {
			sum += col[x];
		}
		return sum;
	}
	
	//Sigmoid function
	float Activation(float input){
		return (float)(1/(Math.pow(2.718f, -input) + 1));
	}
	
	Matrix Activation(Matrix m) {
		Matrix n = new Matrix(m.Rows(), m.Cols());
		for(int x = 0; x<m.Rows(); x++)	{
			for(int y = 0;y<m.Cols(); y++) {
				n.SetCell(Activation(m.GetCell(x, y)), x, y);
			}
		}
		return n;
	}
}