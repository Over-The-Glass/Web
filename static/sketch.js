let speechRec;

function setup() {
    noCanvas();
    speechRec = new p5.SpeechRec('ko-kr', gotSpeech);
    let continuous = true;
    let interimResults = false;
    speechRec.start(continuous, interimResults);
}

function gotSpeech() {
    if (speechRec.resultValue) {
        let said = speechRec.resultString;
        console.log(said); // 한글 자막을 console에 출력
    }
}