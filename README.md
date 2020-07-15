## Hopfield Neural Networks  - code written in the 1980s, used to be in my Java AI book (removed summer of 2020 edition)

Copyright 1989-2020 Mark Watson. All rights reserved.

Hopfield neural networks implement associative (or content addressable) memory. A Hopfield network is trained using a set of patterns and a simple learning algorithm can encode this set of patterns. After training, the network can be shown a pattern similar to one of the training inputs and it will hopefully associate the “noisy” pattern with the correct input pattern. Hopfield networks are very different than back propagation networks (covered later in the [Section of Backpropagation](#backprop)) and deep learning networks (to be covered in the next chapter) because the training data only contains input examples unlike back propagation networks that are trained to associate desired output patterns with input patterns. Internally, the operation of Hopfield neural networks is very different from back propagation networks.

I use Hopfield neural networks to introduce the subject of neural nets because they are very easy to simulate with a program. Hopfield networks are different from most neural architectures that you see today and I hope a brief look at them will convince you that the design space for neural networks and deep learning models is vast with many discoveries and opportunities are still waiting to be explored.

The inputs to Hopfield networks can be any dimensionality. Hopfield networks are often shown as having a two-dimensional input field and are demonstrated recognizing characters, pictures of faces, etc. However, we will lose no generality by implementing a Hopfield neural network toolkit with one-dimensional inputs because a two-dimensional image can be represented by an equivalent one-dimensional array.

How do Hopfield networks work? A simple analogy will help. The trained connection weights in a neural network represent a high dimensional space. This space is folded and convoluted with local minima representing areas around training input patterns. For a moment, visualize this very high dimensional space as just being the three dimensional space inside a room. The floor of this room is a convoluted and curved surface. If you pick up a basketball and bounce it around the room, it will settle at a low point in this curved and convoluted floor. Now, consider that the space of input values is a two-dimensional grid a foot above the floor. For any new input, that is equivalent to a point defined in horizontal coordinates; if we drop our basketball from a position above an input grid point, the basketball will tend to roll down hill into local gravitational minima. The shape of the curved and convoluted floor is a calculated function of a set of training input vectors. After the “floor has been trained” with a set of input vectors, then the operation of dropping the basketball from an input grid point is equivalent to mapping a new input into the training example that is closest to this new input using a neural network.

A common technique in training and using neural networks is to add noise to training data and weights. In the basketball analogy, this is equivalent to “shaking the room” so that the basketball finds a good minima to settle into, and not a non-optimal local minima. We use this technique later when implementing back propagation networks. The weights of back propagation networks are also best visualized as defining a very high dimensional space with a manifold that is very convoluted near areas of local minima. These local minima are centered near the coordinates defined by each input vector.

## Java Classes for Hopfield Neural Networks

The Hopfield neural network model is defined in the file **Hopfield.java** and the file **Hopfield_Test.java** contains the example problem of encoding a set of 1-dimensional bit vectors and then for arbitrary inputs that are not part of the original training data find a matching vector from the training set. Since the file **Hopfield.java** only contains about 65 lines of code, we will look at the code and discuss the algorithms for storing and recall of patterns at the same time. In a Hopfield neural network simulation, every neuron is connected to every other neuron.

Consider a pair of neurons indexed by **i** and **j**. There is a weight **W {i,j}** between these neurons that corresponds in the code to the array element **weight[i,j]**. We can define energy between the associations of these two neurons as:

~~~~~~~~
energy[i,j] = -weight[i,j] * activation[i] * activation[j]
~~~~~~~~

In the Hopfield neural network simulator, we store activations (i.e., the input values) as floating point numbers that get clamped in value to -1 (for off) or +1 (for on). In the energy equation, we consider an activation that is not clamped to a value of one to be zero. This energy is like “gravitational energy potential” using a basketball court analogy: think of a basketball court with an overlaid 2D grid, different grid cells on the floor are at different heights (representing energy
levels) and as you throw a basketball on the court, the ball naturally bounces around and finally stops in a location near to the place you threw the ball, in a low grid cell in the floor - that is, it settles in a locally low energy level. Hopfield networks function in much the same way: when shown a pattern, the network attempts to settle in a local minimum energy point as defined by a previously seen training example.

When training a network with a new input, we are looking for a low energy point near the new input vector. The total energy is a sum of the above equation over all (i,j).

The class constructor allocates storage for input values, temporary storage, and a two-dimensional array to store weights:

~~~~~~~~
     public Hopfield(int numInputs) {
       this.numInputs = numInputs;
       weights = new float[numInputs][numInputs];
       inputCells = new float[numInputs];
       tempStorage = new float[numInputs];
     } 
~~~~~~~~

Remember that this model is general purpose: multi-dimensional inputs can be converted to an equivalent one-dimensional array. The method **addTrainingData** is used to store an input data array for later training. All input values get clamped to an “off” or “on” value by the utility method **adjustInput**. The utility method **truncate** truncates floating-point values to an integer value. The utility method **deltaEnergy** has one argument: an index into the input vector. The class variable **tempStorage** is set during training to be the sum of a row of trained weights. So, the method **deltaEnergy** returns a measure of the energy difference between the input vector in the current input cells and the training input examples:

~~~~~~~~
     private float deltaEnergy(int index) {
       float temp = 0.0f;
       for (int j=0; j<numInputs; j++) {
         temp += weights[index][j] * inputCells[j];
       }
       return 2.0f * temp - tempStorage[index];
     } 
~~~~~~~~

The method **train** is used to set the two-dimensional weight array and the one-dimensional **tempStorage** array in which each element is the sum of the corresponding row in the two-dimensional weight array:

~~~~~~~~
     public void train() {
       for (int j=1; j<numInputs; j++) {
         for (int i=0; i<j; i++) {
           for (int n=0; n<trainingData.size(); n++) {
             float [] data = (float [])trainingData.elementAt(n);
             float temp1 = adjustInput(data[i]) * adjustInput(data[j]);
             float temp = truncate(temp1 + weights[j][i]);
             weights[i][j] = weights[j][i] = temp;
           }
         }
       }
       for (int i=0; i<numInputs; i++) {
         tempStorage[i] = 0.0f;
         for (int j=0; j<i; j++) {
           tempStorage[i] += weights[i][j];
         }
       }
     } 
~~~~~~~~

Once the arrays **weight** and **tempStorage** are defined, it is simple to recall an original input pattern from a similar test pattern:

~~~~~~~~
     public float [] recall(float [] pattern, int numIterations) {
       for (int i=0; i<numInputs; i++) {
         inputCells[i] = pattern[i];
       }
       for (int ii = 0; ii<numIterations; ii++) {
         for (int i=0; i<numInputs; i++) {
           if (deltaEnergy(i) > 0.0f) {
             inputCells[i] = 1.0f;
           } else {
             inputCells[i] = 0.0f;
           }
         }
       }
       return inputCells;
     } 
~~~~~~~~

## Testing the Hopfield Neural Network Class

The test program for the Hopfield neural network class is
**Test\_Hopfield**. This test program defined three test input patterns, each with ten values:

~~~~~~~~
     static float [] data [] = {
       { 1, 1, 1, -1, -1, -1, -1, -1, -1, -1},
       {-1, -1, -1, 1, 1, 1, -1, -1, -1, -1},
       {-1, -1, -1, -1, -1, -1, -1, 1, 1, 1} }; 
~~~~~~~~

The following code fragment shows how to create a new instance of the **Hopfield** class and train it to recognize these three test input patterns:

~~~~~~~~
     test = new Hopfield(10);
     test.addTrainingData(data[0]);
     test.addTrainingData(data[1]);
     test.addTrainingData(data[2]);
     test.train(); 
~~~~~~~~

The static method **helper** is used to slightly scramble an input pattern, then test the training Hopfield neural network to see if the original pattern is re-created:

~~~~~~~~
     helper(test, "pattern 0", data[0]);
     helper(test, "pattern 1", data[1]);
     helper(test, "pattern 2", data[2]); 
~~~~~~~~

The following listing shows an implementation of the method **helper** (the called method **pp** simply formats a floating point number for printing by clamping it to zero or one). This version of the code randomly flips one test bit and we will see that the trained Hopfield network almost always correctly recognizes the original pattern. The version of method **helper** included in the ZIP file for this book is slightly different in that two bits are randomly flipped (we will later look at sample output with both one and two bits randomly flipped).

~~~~~~~~
     private static void helper(Hopfield test,
                                String s,
                                float [] test_data) {
       float [] dd = new float[10];
       for (int i=0; i<10; i++) {
         dd[i] = test_data[i];
       }
       int index = (int)(9.0f * (float)Math.random());
       if (dd[index] < 0.0f) dd[index] = 1.0f; else dd[index] = -1.0f;
       float [] rr = test.recall(dd, 5);
       System.out.print(s+"\nOriginal data: ");
       for (int i = 0; i < 10; i++)
         System.out.print(pp(test_data[i]) + " ");
       System.out.print("\nRandomized data: ");
       for (int i = 0; i < 10; i++)
         System.out.print(pp(dd[i]) + " ");
       System.out.print("\nRecognized pattern: ");
       for (int i = 0; i < 10; i++)
         System.out.print(pp(rr[i]) + " ");
       System.out.println();
     } 
~~~~~~~~

There is a *Makefile* that has targets for running all of the examples in this chapter. You can also run this example using:

~~~~~~~~
mvn install -DskipTests
mvn test
~~~~~~~~


The following listing shows how to run the program, and lists the example output:

~~~~~~~~
     pattern 0
     Original data: 1 1 1 0 0 0 0 0 0 0
     Randomized data: 1 1 1 0 0 0 1 0 0 0
     Recognized pattern: 1 1 1 0 0 0 0 0 0 0
     pattern 1
     Original data: 0 0 0 1 1 1 0 0 0 0
     Randomized data: 1 0 0 1 1 1 0 0 0 0
     Recognized pattern: 0 0 0 1 1 1 0 0 0 0
     pattern 2
     Original data: 0 0 0 0 0 0 0 1 1 1
     Randomized data: 0 0 0 1 0 0 0 1 1 1
     Recognized pattern: 0 0 0 0 0 0 0 1 1 1 
~~~~~~~~

In this listing we see that the three sample training patterns in **Test\_Hopfield.java** are re-created after scrambling the data by changing one randomly chosen value to its opposite value. When you run the test program several times you will see occasional errors when one
random bit is flipped and you will see errors occur more often with two bits flipped. Here is an example with two bits flipped per test: the first pattern is incorrectly reconstructed and the second and third patterns are reconstructed correctly:

~~~~~~~~
     pattern 0
     Original data: 1 1 1 0 0 0 0 0 0 0
     Randomized data: 0 1 1 0 1 0 0 0 0 0
     Recognized pattern: 1 1 1 1 1 1 1 0 0 0
     pattern 1
     Original data: 0 0 0 1 1 1 0 0 0 0
     Randomized data: 0 0 0 1 1 1 1 0 1 0
     Recognized pattern: 0 0 0 1 1 1 0 0 0 0
     pattern 2
     Original data: 0 0 0 0 0 0 0 1 1 1
     Randomized data: 0 0 0 0 0 0 1 1 0 1
     Recognized pattern: 0 0 0 0 0 0 0 1 1 1 
~~~~~~~~

