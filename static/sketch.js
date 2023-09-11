let speechRec = new p5.SpeechRec('ko-kr', gotSpeech);
let outputText = "";
let continuous = true;
let interimResults = true;
let said = ''; // 음성 결과를 저장하는 변수
let speakerName = '';

function setup() {
    noCanvas();
    outputTextElement = document.getElementById('outputText');
    speechRec.onEnd = restart;
    speechRec.start(continuous, interimResults);
    // setInterval(gotSpeech, 100); // 일정 시간마다 자막 업데이트 호출
}

function gotSpeech() {
    if (speechRec.resultValue) {
        said = speechRec.resultString;
        fetch('/speaker_info')
            .then(response => response.json())
            .then(data => {
                speakerName = data.speaker;
            })
            .catch(error => {
                console.log('Error:', error);
            });
        outputTextElement.innerHTML = speakerName + "," + said;
        var jsonData = {
            flag: 'in conversation mode',
            change: 'subtitle',
            speaker: speakerName,
            subtitle: said
        };
        Unity.call(JSON.stringify(jsonData));
    }
    console.log(speechRec.resultJSON.results);
    // console.log(speechRec.resultJSON.results[0][0]);
    // console.log(speechRec.resultJSON.results[0].isFinal);
    // console.log(speechRec.resultJSON.results[0][0].transcript);
    // console.log("자막: " + speechRec.resultString + ", isFinal: " + speechRec.resultJSON.results[0].isFinal);
    
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

function restart() {
    speechRec.start(continuous, interimResults);
}