import neuralnetwork.network.*;
NeuralNetwork nn = new NeuralNetwork(4, new int[]{784,16,16,10});

void setup(){
  background(0);
  size(200,200);
  textAlign(CENTER,CENTER);
  text(nn.toString(), 0,0,width,height);
}