/* ============================================================
   SiVoice shared.js
   MediaPipe + Flask AI + Gesture Rules
   ============================================================ */

const AI_INTERVAL = 500;
const MIN_CONFIDENCE = 0.80;

let lastPrediction = "";
let lastPredictionConfidence = 0;
let lastPredictionTime = 0;
let predictionBusy = false;


/* ============================================================
   SIGNS
   ============================================================ */

const SIGNS = [
    {
        name: "A",
        description: "Make the A hand shape."
    },
    {
        name: "B",
        description: "Make the B hand shape."
    },
    {
        name: "C",
        description: "Make the C hand shape."
    },
    {
        name: "D",
        description: "Make the D hand shape."
    },
    {
        name: "Open Palm",
        description: "Open all five fingers."
    },
    {
        name: "Peace",
        description: "Raise index and middle fingers."
    },
    {
        name: "Thumbs Up",
        description: "Raise your thumb upward."
    },
    {
        name: "Welcome",
        description: "Open your palm and wave it left and right."
    }
];

/* Make SIGNS available to inline scripts in index.html */
window.SIGNS = SIGNS;


/* ============================================================
   SPEECH
   ============================================================ */

let lastSpoken = "";

function speakGesture(text) {

    if (!text) return;

    if (!("speechSynthesis" in window)) return;

    if (text === lastSpoken) return;

    lastSpoken = text;

    window.speechSynthesis.cancel();

    const speech =
        new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 0.9;
    speech.pitch = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);
}


/* ============================================================
   LANDMARK HELPERS
   ============================================================ */

/*
MediaPipe hand landmark indexes:

0  = Wrist

Thumb:
1  = Thumb CMC
2  = Thumb MCP
3  = Thumb IP
4  = Thumb tip

Index:
5  = Index MCP
6  = Index PIP
7  = Index DIP
8  = Index tip

Middle:
9  = Middle MCP
10 = Middle PIP
11 = Middle DIP
12 = Middle tip

Ring:
13 = Ring MCP
14 = Ring PIP
15 = Ring DIP
16 = Ring tip

Pinky:
17 = Pinky MCP
18 = Pinky PIP
19 = Pinky DIP
20 = Pinky tip
*/


function distance(a, b) {

    const dx = a.x - b.x;
    const dy = a.y - b.y;

    return Math.sqrt(
        dx * dx +
        dy * dy
    );
}


function fingerExtended(
    landmarks,
    tip,
    pip,
    mcp
) {

    if (!landmarks[tip] ||
        !landmarks[pip] ||
        !landmarks[mcp]) {

        return false;
    }

    /*
       In MediaPipe coordinates,
       smaller Y means higher on screen.
    */

    return (
        landmarks[tip].y <
        landmarks[pip].y &&
        landmarks[pip].y <=
        landmarks[mcp].y
    );
}


function thumbExtended(landmarks) {

    if (!landmarks[4] ||
        !landmarks[3] ||
        !landmarks[2]) {

        return false;
    }

    /*
       For the thumb, use distance from wrist
       because thumb orientation changes.
    */

    const wrist = landmarks[0];

    const tipDistance =
        distance(landmarks[4], wrist);

    const mcpDistance =
        distance(landmarks[2], wrist);

    return tipDistance >
           mcpDistance * 1.25;
}


function getFingerState(landmarks) {

    return {

        thumb:
            thumbExtended(landmarks),

        index:
            fingerExtended(
                landmarks,
                8,
                6,
                5
            ),

        middle:
            fingerExtended(
                landmarks,
                12,
                10,
                9
            ),

        ring:
            fingerExtended(
                landmarks,
                16,
                14,
                13
            ),

        pinky:
            fingerExtended(
                landmarks,
                20,
                18,
                17
            )
    };
}


/* ============================================================
   OPEN PALM
   ============================================================ */

function isOpenPalm(landmarks) {

    const fingers =
        getFingerState(landmarks);

    let extendedCount = 0;

    if (fingers.thumb) extendedCount++;
    if (fingers.index) extendedCount++;
    if (fingers.middle) extendedCount++;
    if (fingers.ring) extendedCount++;
    if (fingers.pinky) extendedCount++;

    return extendedCount >= 4;
}


/* ============================================================
   PEACE
   ============================================================ */

function isPeace(landmarks) {

    const fingers =
        getFingerState(landmarks);

    /*
       Index + middle extended.
       Ring + pinky folded.
    */

    return (
        fingers.index &&
        fingers.middle &&
        !fingers.ring &&
        !fingers.pinky
    );
}


/* ============================================================
   THUMBS UP
   ============================================================ */

function isThumbsUp(landmarks) {

    const fingers =
        getFingerState(landmarks);

    /*
       Thumb extended.
       Other four fingers folded.
    */

    if (!fingers.thumb) {
        return false;
    }

    if (fingers.index ||
        fingers.middle ||
        fingers.ring ||
        fingers.pinky) {

        return false;
    }

    /*
       Thumb should point upward.
    */

    const thumbTip =
        landmarks[4];

    const thumbMcp =
        landmarks[2];

    if (!thumbTip ||
        !thumbMcp) {

        return false;
    }

    return (
        thumbTip.y <
        thumbMcp.y
    );
}


/* ============================================================
   WELCOME / WAVE DETECTION
   ============================================================ */

let waveHistory = [];

const WAVE_HISTORY_SIZE = 18;
const WAVE_MIN_MOVEMENT = 0.10;


function getPalmCenter(landmarks) {

    const wrist = landmarks[0];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];
    const pinkyMcp = landmarks[17];

    return {

        x:
            (
                wrist.x +
                indexMcp.x +
                middleMcp.x +
                ringMcp.x +
                pinkyMcp.x
            ) / 5,

        y:
            (
                wrist.y +
                indexMcp.y +
                middleMcp.y +
                ringMcp.y +
                pinkyMcp.y
            ) / 5
    };
}


function detectWelcome(landmarks) {

    /*
       Welcome = open palm + horizontal movement.
    */

    if (!isOpenPalm(landmarks)) {

        waveHistory = [];

        return false;
    }

    const center =
        getPalmCenter(landmarks);

    waveHistory.push({
        x: center.x,
        y: center.y,
        time: Date.now()
    });

    if (
        waveHistory.length >
        WAVE_HISTORY_SIZE
    ) {

        waveHistory.shift();
    }

    if (
        waveHistory.length <
        8
    ) {

        return false;
    }

    let minX = Infinity;
    let maxX = -Infinity;

    for (
        const point of waveHistory
    ) {

        minX =
            Math.min(
                minX,
                point.x
            );

        maxX =
            Math.max(
                maxX,
                point.x
            );
    }

    const movement =
        maxX - minX;

    return movement >= WAVE_MIN_MOVEMENT;
}


/* ============================================================
   RULE-BASED GESTURE CLASSIFIER
   ============================================================ */

function classifyGesture(landmarks) {

    if (!landmarks) {

        return {
            label: "No Sign",
            confidence: 0
        };
    }


    /* --------------------------------------------------------
       WELCOME
       -------------------------------------------------------- */

    if (detectWelcome(landmarks)) {

        return {
            label: "Welcome",
            confidence: 0.95
        };
    }


    /* --------------------------------------------------------
       THUMBS UP
       -------------------------------------------------------- */

    if (isThumbsUp(landmarks)) {

        return {
            label: "Thumbs Up",
            confidence: 0.95
        };
    }


    /* --------------------------------------------------------
       PEACE
       -------------------------------------------------------- */

    if (isPeace(landmarks)) {

        return {
            label: "Peace",
            confidence: 0.95
        };
    }


    /* --------------------------------------------------------
       OPEN PALM
       -------------------------------------------------------- */

    if (isOpenPalm(landmarks)) {

        return {
            label: "Open Palm",
            confidence: 0.95
        };
    }


    /*
       Nothing matched.
       Let TensorFlow handle A-Z.
    */

    return null;
}


/* ============================================================
   AI PREDICTION
   ============================================================ */

async function predictWithAI(video) {

    if (!video) return;

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return;
    }

    const now = Date.now();

    if (
        now - lastPredictionTime <
        AI_INTERVAL
    ) {
        return;
    }

    if (predictionBusy) {
        return;
    }

    lastPredictionTime = now;
    predictionBusy = true;

    try {

        const tempCanvas =
            document.createElement("canvas");

        tempCanvas.width = 224;
        tempCanvas.height = 224;

        const ctx =
            tempCanvas.getContext("2d");

        ctx.drawImage(
            video,
            0,
            0,
            224,
            224
        );

        const blob =
            await new Promise(resolve => {

                tempCanvas.toBlob(
                    resolve,
                    "image/jpeg",
                    0.85
                );

            });

        if (!blob) {
            return;
        }

        const formData =
            new FormData();

        formData.append(
            "image",
            blob,
            "frame.jpg"
        );

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );

        if (!response.ok) {

            throw new Error(
                "Prediction server error: " +
                response.status
            );
        }

        const data =
            await response.json();

        lastPrediction =
            String(
                data.letter ||
                "No Sign"
            );

        lastPredictionConfidence =
            Number(
                data.confidence ||
                0
            );

        console.log(
            "AI:",
            lastPrediction,
            lastPredictionConfidence + "%"
        );

    }
    catch (error) {

        console.error(
            "AI prediction failed:",
            error
        );

    }
    finally {

        predictionBusy = false;

    }
}


/* ============================================================
   HAND RECOGNIZER
   ============================================================ */

class HandRecognizer {

    constructor(video, canvas) {

        this.video = video;

        this.canvas = canvas;

        this.ctx =
            canvas.getContext("2d");

        this.callback = null;

        this.hands = null;

        this.camera = null;

        this.running = false;

        this.currentGesture = null;
    }


    onResult(callback) {

        this.callback = callback;

    }


    async start() {

        if (this.running) {
            return;
        }


        /* ----------------------------------------------------
           Check MediaPipe
        ---------------------------------------------------- */

        if (typeof Hands === "undefined") {

            throw new Error(
                "MediaPipe Hands not loaded"
            );

        }

        if (typeof Camera === "undefined") {

            throw new Error(
                "MediaPipe Camera not loaded"
            );

        }


        /* ----------------------------------------------------
           Create MediaPipe Hands
        ---------------------------------------------------- */

        this.hands =
            new Hands({

                locateFile: function(file) {

                    return (
                        "https://cdn.jsdelivr.net/npm/" +
                        "@mediapipe/hands/" +
                        file
                    );

                }

            });


        this.hands.setOptions({

            maxNumHands: 2,

            modelComplexity: 1,

            minDetectionConfidence: 0.6,

            minTrackingConfidence: 0.6

        });


        /* ----------------------------------------------------
           MediaPipe results
        ---------------------------------------------------- */

        this.hands.onResults(
            (results) => {

                const ctx =
                    this.ctx;

                const canvas =
                    this.canvas;


                /* ------------------------------------------------
                   Clear canvas
                ------------------------------------------------ */

                ctx.save();

                ctx.clearRect(
                    0,
                    0,
                    canvas.width,
                    canvas.height
                );


                /* ------------------------------------------------
                   Draw detected hands
                ------------------------------------------------ */

                if (
                    results.multiHandLandmarks &&
                    results.multiHandLandmarks.length > 0
                ) {

                    for (
                        const landmarks
                        of results.multiHandLandmarks
                    ) {

                        drawConnectors(
                            ctx,
                            landmarks,
                            HAND_CONNECTIONS,
                            {
                                color: "#00FFFF",
                                lineWidth: 3
                            }
                        );

                        drawLandmarks(
                            ctx,
                            landmarks,
                            {
                                color: "#00FF00",
                                radius: 4
                            }
                        );

                    }

                }

                ctx.restore();


                /* ------------------------------------------------
                   No callback
                ------------------------------------------------ */

                if (!this.callback) {
                    return;
                }


                /* ------------------------------------------------
                   No hand
                ------------------------------------------------ */

                if (
                    !results.multiHandLandmarks ||
                    results.multiHandLandmarks.length === 0
                ) {

                    this.currentGesture = null;

                    this.callback({

                        label: "No Hand",

                        confidence: 0

                    });

                    return;
                }


                /* ------------------------------------------------
                   Rule-based gesture recognition
                ------------------------------------------------ */

                const landmarks =
                    results.multiHandLandmarks[0];

                const gesture =
                    classifyGesture(
                        landmarks
                    );


                /* ------------------------------------------------
                   If one of our four gestures is detected
                ------------------------------------------------ */

                if (gesture) {

                    this.currentGesture =
                        gesture.label;

                    this.callback({

                        label:
                            gesture.label,

                        confidence:
                            gesture.confidence

                    });

                    return;
                }


                /* ------------------------------------------------
                   Otherwise use TensorFlow A-Z
                ------------------------------------------------ */

                if (
                    lastPrediction &&
                    lastPrediction !== "No Sign"
                ) {

                    this.callback({

                        label:
                            lastPrediction,

                        confidence:
                            lastPredictionConfidence /
                            100

                    });

                }
                else {

                    this.callback({

                        label:
                            "Detecting...",

                        confidence:
                            0

                    });

                }

            }
        );


        /* ----------------------------------------------------
           CAMERA
        ---------------------------------------------------- */

        this.camera =
            new Camera(

                this.video,

                {

                    onFrame:
                        async () => {

                            /*
                               Send frame to MediaPipe
                            */

                            await this.hands.send({

                                image:
                                    this.video

                            });


                            /*
                               Send frame to Flask AI
                            */

                            predictWithAI(
                                this.video
                            );

                        },

                    width: 640,

                    height: 480

                }

            );


        /* ----------------------------------------------------
           Canvas size
        ---------------------------------------------------- */

        this.canvas.width = 640;
        this.canvas.height = 480;


        /* ----------------------------------------------------
           Start camera
        ---------------------------------------------------- */

        this.camera.start();

        this.running = true;

        console.log(
            "Camera started successfully"
        );

    }

}


/* ============================================================
   GLOBAL EXPORT
   ============================================================ */

window.HandRecognizer =
    HandRecognizer;

window.speakGesture =
    speakGesture;


/* ============================================================
   DEBUG
   ============================================================ */

console.log(
    "SiVoice shared.js loaded"
);

console.log(
    "MediaPipe:",
    typeof Hands !== "undefined"
);

console.log(
    "Camera:",
    typeof Camera !== "undefined"
);

console.log(
    "HandRecognizer:",
    typeof HandRecognizer !== "undefined"
);

console.log(
    "Gesture rules: Open Palm / Peace / Thumbs Up / Welcome"
);
