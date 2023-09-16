function login_() {
    var em = document.querySelector('#em');
    var pw = document.querySelector('#pw');

    if (em.value == "" || pw.value == "") {
        alert("회원가입을 할 수 없습니다.");
    } else {
        location.href = 'menu.html';
    }
}