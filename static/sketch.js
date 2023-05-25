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
        console.log(said);

        // Ajax 요청을 보내고 발화자 정보와 한국어 자막을 가져옴
        fetch('/speaker_info')
            .then(response => response.json())
            .then(data => {
                console.log('Speaker:', data.speaker);
            })
            .catch(error => {
                console.log('Error:', error);
            });
    }
}