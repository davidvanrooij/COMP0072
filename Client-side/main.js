var ctx = document.getElementById('canvas').getContext('2d');

var init = true;

// Function to show welcome text
function show_welcome_text() {
  ctx.font = '20px Noto Sans';
  ctx.fillText('Start drawing symbols here!', 100, 150-10);
}

// Clear canvas
function clear_canvas(){
	ctx.clearRect(0,0,canvas.width, canvas.height);
}

// Let the user draw lines
function draw_line(event) {
  var x = event.layerX;
  var y = event.layerY;
  ctx.beginPath();
	ctx.arc(x,y,10,0,2*Math.PI);
	ctx.fill();

}

show_welcome_text();



canvas.addEventListener('mousedown', function(){

	// Remove welcome text the first time you click on the canvas
	if(init){
		clear_canvas();
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
