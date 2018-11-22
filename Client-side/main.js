var ctx = document.getElementById('canvas').getContext('2d');

var init = true;

function show_welcome_text() {
  ctx.font = '20px Noto Sans';
  ctx.fillText('Start drawing symbols here!', 100, 150-10);
}

show_welcome_text();

function draw_line(event) {
  var x = event.layerX;
  var y = event.layerY;
  ctx.beginPath();
	ctx.arc(x,y,10,0,2*Math.PI);
	ctx.fill();

}

canvas.addEventListener('mousedown', function(){

	// Remove welcome text the first time you click on the canvas
	if(init){
		ctx.clearRect(0,0,canvas.width, canvas.height);
		init = false;
	}

	canvas.addEventListener('mousemove', function(event){
		draw_line(event);
	});
});

// canvas.addEventListener('mouseup', function(event){
// 	canvas.removeEventListener('mousedown');
// })

dataURL = canvas.toDataURL();
