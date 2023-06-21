let speechRec = new p5.SpeechRec('en-us', gotSpeech);

let continuous = true;
let interimResults = false;
let said = ''; // 음성 결과를 저장하는 변수

function gotSpeech() {
    if (speechRec.resultValue) {
        said = speechRec.resultString;
    }
}

function setup() {
    noCanvas();
    speechRec.start(continuous, interimResults);
    setInterval(updateCaption, 100); // 일정 시간마다 자막 업데이트 호출
}

function updateCaption() {
    if (said !== '') {
        $.ajax({
            url: '/process_speech',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ 'speech': said }),
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        });
        said = ''; // 자막 업데이트 후 초기화
    }
}