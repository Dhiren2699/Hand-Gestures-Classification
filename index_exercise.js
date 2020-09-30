// let for block scope, they can be shared accross other functions

let mobilenet;
let model;

//for webcam, webcame class is stored in webcam.js
const webcam = new Webcam(document.getElementById('wc'));

//declare rps-dataset.js class
const dataset = new RPSDataset();

var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0;
let isPredicting = false;

//loading trained mobilenet model
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  //get output layer of preloaded model
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  //new model
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  //one-hot encode labels in dataset (when we train we use ys)
  dataset.ys = null;
  dataset.encodeLabels(5);
  
  //input of flatten is output of truncated mobiles net
  model = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape:mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({units:100,activation:'relu'}),
        tf.layers.dense({units:5,activation:'softmax'})
    ]
  });
    
   
  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(lr=0.0001);
    
        
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({optimizer:optimizer,loss:'categoricalCrossentropy'});
 
  let loss = 0;
  await model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;  
		case "3":
			spockSamples++;
			document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
			break;

    case "4":
        lizardSamples++;
        document.getElementById("lizardsamples").innerHTML = "Lizard Samples:" + lizardSamples;
        break;
		
            
  }
  //extract label from id by converting it to int
  label = parseInt(elem.id);
  
  //capture contents of webcame to extract features
  const img = webcam.capture();
  

	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  //true when clicked predict
  while (isPredicting) {

    //read the frame from webcam
    //tidy to prevent memory leaks
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      //argMax to return a 1D tensor containing prediction
      return predictions.as1D().argMax();
    });

    //update ui
    // get classid from what webcam sees
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
		case 3:
			predictionText = "I see Spock";
			break;
            
    case 4:
        predictionText = "I see Lizard";
        break;
	
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    //this prevents us from locking ui thread so page can stay responsive
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


async function init(){
  //setup webcam
  await webcam.setup();
  
  //load model
  mobilenet = await loadMobilenet();
  
  //initialize model
  //tf.tidy throws away unneeded tensors so they dont hang around memory
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();