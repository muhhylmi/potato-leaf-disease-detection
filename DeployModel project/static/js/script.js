var jenisDaun =  $("#jenisDaun").html();

if (jenisDaun!="") {
	$("#fiturPrediksi").css("visibility", "all");
} else {
	$("#fiturPrediksi").css("visibility", "hidden");
}

var cardjenis = document.querySelector('#jenisdauncard');
var buttonprediksi = document.querySelector('#btnpredik');
var loader = document.querySelector('#loader');

console.log(buttonprediksi);

buttonprediksi.addEventListener('click',()=>{
	loader.style.display='inline';
});

