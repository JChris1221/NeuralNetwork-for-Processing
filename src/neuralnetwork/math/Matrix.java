package neuralnetwork.math;

//import processing.core.*;
import java.lang.Math;
import neuralnetwork.exceptions.*;
/**
 *The matrix class is used for creating matrix objects and perform
 *basic matrix arithmetic operations
 *
 *The idea referenced from the coding train
 *https://thecodingtrain.com/
 */

public class Matrix {
	public float[][] mat; // the matrix represented in 2d array
	
	//CONSTRUCTORS
	public Matrix(int rows, int cols){
	 this.mat = new float[rows][cols];
	}
	
	public Matrix(float[][] mat_){
		this.mat = mat_.clone();
	}
	
	public Matrix(float[] mat_) {
		this.mat = new float[][] {mat_.clone()};
	}
	
	public Matrix(float[] mat_, boolean isColumned) {
		if(isColumned){
			this.mat = new float[mat_.length][1];
			for(int x = 0; x<mat_.length;x++) {
				this.mat[x][0] = mat_[x];
			}
		}
		else
			this.mat = new float[][] {mat_.clone()};
	}
	//END CONSTRUCTORS
	
	//Returns the 2d Array row size
	public int Rows(){
		return mat.length;
	}
	
	//Returns the 2d array column size
	public int Cols(){
		return mat[0].length;
	}
	
	//Generates a random value between 0 and 1
	public void Randomize(){
		for(int x = 0; x<mat.length; x++){
			for(int y = 0; y < mat[x].length; y++) {
				this.mat[x][y] = (float)Math.random();
			}
		}
	}
	
	public void Randomize(float variant){
		for(int x = 0; x<mat.length; x++){
			for(int y = 0; y < mat[x].length; y++) {
				this.mat[x][y] = (float)Math.random()*(float)Math.sqrt(variant);
			}
		}
	}
	
	
	//Adds the input matrix
	public void Add(Matrix b)
	{
		if(this.Rows() != b.Rows() || this.Cols() != b.Cols()) {
			String m = "Matrices do not match dimensions";
			throw new MatrixArithmeticException(m);
		}
			
		for(int x = 0; x < this.Rows(); x++) {
			for (int y = 0; y < this.Cols(); y++) {
				this.mat[x][y] += b.mat[x][y];
			}
		}
	}
	
	//Returns a string version of the 2d array (for debugging)
	public String toString(){
		String s = new String();
		for(int x = 0; x<this.mat.length; x++) {
			for(int y = 0; y<this.mat[0].length; y++) {
				s+= " " + mat[x][y] + ",";
			}
			s+="\n";
		}
		return s;
	}
	
	//Subtracts the input matrix to this instance
	public void Sub(Matrix b)
	{
		if(this.Rows() != b.Rows() || this.Cols() != b.Cols()) {
			String m = "Matrices do not match dimensions";
			throw new MatrixArithmeticException(m);
		}
		
		for(int x = 0; x < this.Rows(); x++) {
			for (int y = 0; y < this.Cols(); y++) {
				this.mat[x][y] -= b.mat[x][y];
			}
		}
	}
	
	//Get the hadamard product of this matrix and matrix argument
	public void HadamardProduct(Matrix b) {
		if(this.Rows() != b.Rows() || this.Cols() != b.Cols()) {
			String m = "Matrices do not match dimensions";
			throw new MatrixArithmeticException(m);
		}
		
		for(int x = 0; x < this.Rows(); x++) {
			for(int y = 0; y<this.Cols(); y++) {
				this.mat[x][y] *= b.GetCell(x, y);
			}
		}
	}
	
	//Changes rows into columns
	public void Transpose() {
		Matrix trans = new Matrix(this.Cols(), this.Rows());
		for(int row = 0; row < trans.Rows(); row++) {
			trans.mat[row] = this.GetColumn(row);
		}
	}
	
	//Adds constant value to the matrix
	public void ScalarAdd(float val) {
		for(int x = 0; x < this.Rows(); x++) {
			for(int y = 0; y<this.Cols(); y++) {
				this.mat[x][y] += val;
			}
		}
	}
	
	//Subtracts constant value to the matrix
	public void ScalarSub(float val) {
		for(int x = 0; x < this.Rows(); x++) {
			for(int y = 0; y<this.Cols(); y++) {
				this.mat[x][y] -= val;
			}
		}
	}
	
	//Multiply matrix by constant
	public void ScalarMul(float val) {
		for(int x = 0; x < this.Rows(); x++) {
			for(int y = 0; y<this.Cols(); y++) {
				this.mat[x][y] *= val;
			}
		}
	}
	
	
	//Multiply matrices and return as a new matrix
	public static Matrix Mul(Matrix a, Matrix b)
	{
		if(a.Cols() != b.Rows()) {
			String m = "Matrices do not have valid dimensions to perform multiplication";
			throw new MatrixArithmeticException(m);
		}
		Matrix mul = new Matrix(a.Rows(),b.Cols());
		for(int x = 0; x < a.Rows(); x++){
			for (int y = 0; y < b.Cols(); y++){
				for (int z = 0; z < a.Cols(); z++){
					mul.mat[x][y] += a.mat[x][z] * b.mat[z][y]; 
				}
			}
		}
		
		return mul;
	}
	
	
	//Returns the 2d array of the matrix class
	public float[][] GetMatrix(){
		return this.mat.clone();
	}
	
	//returns a column on the 2d array
	public float[] GetColumn(int index) {
		float[] col = new float[this.mat.length];
		for(int x = 0; x<this.mat.length; x++) {
			col[x] = this.mat[x][index];
		}
		return col;
	}
	
	//Returns a row on the 2d array
	public float[] GetRow(int layer){
		return this.mat[layer].clone();
	}
	
	//Change a value in the 2d array with the indeces given
	public void SetCell(float val, int r, int c) {
		this.mat[r][c] = val;
	}
	
	//Return the value of the 2d array on the given indeces
	public float GetCell(int r, int c) {
		return this.mat[r][c];
	}
	
	//Returns the 2d array values
//	public float[][] toArray(){
//		return this.mat;
//	}
}