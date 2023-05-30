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
        let speaker = "";
        fetch('/speaker_info')
            .then(response => response.json())
            .then(data => {
                speaker = data.speaker;
                console.log(speaker + ": " + said);
            })
            .catch(error => {
                console.log('Error:', error);
            });
    }
}