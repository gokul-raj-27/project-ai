let lastSpoken = "";

function speakGesture(text) {

    if (text === lastSpoken) return;

    lastSpoken = text;

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 1;
    speech.pitch = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);
}

let lastPrediction = "";
let lastPredictionTime = 0;

async function predictWithAI(video){

    const now = Date.now();

    // Predict once every 500ms
    if(now - lastPredictionTime < 500){
        return;
    }

    lastPredictionTime = now;

    const canvas = document.createElement("canvas");

    canvas.width = 224;
    canvas.height = 224;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video,0,0,224,224);

    canvas.toBlob(async(blob)=>{

        const formData = new FormData();

        formData.append("image",blob,"frame.jpg");

        try{

            const response = await fetch("http://127.0.0.1:5000/predict",{

                method:"POST",

                body:formData

            });

            const data = await response.json();

            lastPrediction = data.letter;

            if (data.letter !== "No Sign") {
              speakGesture(data.letter);
}

        }

        catch(err){

            console.log(err);

        }

    },"image/jpeg");

}

class HandRecognizer {

  constructor(video, canvas) {

    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    this.callback = null;

    this.hands = null;
    this.camera = null;

  }

  onResult(cb) {
    this.callback = cb;
  }

  async start() {

    this.hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6
    });

    this.hands.onResults((results) => {

      const ctx = this.ctx;
      const canvas = this.canvas;

      ctx.save();

      ctx.clearRect(0,0,canvas.width,canvas.height);

      if(results.multiHandLandmarks){
        if (results.multiHandedness) {
        results.multiHandedness.forEach((hand, index) => {
            console.log("Hand", index + 1, hand.label);
        });
    }

        for(const landmarks of results.multiHandLandmarks){

          drawConnectors(
            ctx,
            landmarks,
            HAND_CONNECTIONS,
            {
                color:"#00FFFF",
                lineWidth:4
            }
        );

        drawLandmarks(
            ctx,
            landmarks,
            {
                color:"#00FF00",
                radius:5
            }
        );

    }

}

      ctx.restore();

      if(this.callback){
        let output = {
        label: "No Hand",
        confidence: 0
      };

      if (results.multiHandLandmarks &&
    results.multiHandLandmarks.length > 0) {

    if (lastPrediction === "") {
    output.label = "Detecting...";
} else {
    output.label = lastPrediction;
}
    output.confidence = 0.90;
}
      this.callback(output);

      }

    });

    this.camera = new Camera(this.video,{

      onFrame: async ()=>{

        await this.hands.send({
          image:this.video
        });
        await predictWithAI(this.video);

      },

      width:640,
      height:480

    });

    this.canvas.width = 640;
    this.canvas.height = 480;

    this.camera.start();

  }

}