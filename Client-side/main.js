var ctx = document.getElementById('canvas').getContext('2d');

var init = true;
var mouse_pressed = false;
var x_0, y_0 = (0, 0);


// Function to show welcome text
function show_welcome_text() {
	ctx.font = '20px Noto Sans';
	ctx.fillText('Start drawing symbols here!', 100, 150-10);
}

// Clear canvas
function clear_canvas(){
	ctx.clearRect(0,0,canvas.width, canvas.height);
}

function create_border(){
	// top
	ctx.clearRect(0,0, canvas.width, 10);

	// left
	ctx.clearRect(0,0, 10, canvas.height);

	// right
	ctx.clearRect(canvas.width - 10, 0, 10, canvas.height);

	// bottom
	ctx.clearRect(0, canvas.height - 10, canvas.width, 10);
}

// Let the user draw lines
function draw_line(event) {
	// Get curser coordinates 
	var x_1 = event.layerX;
	var y_1 = event.layerY;

	ctx.beginPath();
	ctx.moveTo(x_0,y_0);
	ctx.lineTo(x_1,y_1);
	ctx.lineWidth = 10;
	ctx.lineJoin = 'round';
	ctx.closePath();
	ctx.stroke();

	// Save curser coordinates as last point
	x_0 = x_1;
	y_0 = y_1;
}

show_welcome_text();


canvas.addEventListener('mousemove', function(event){
	if(mouse_pressed){
		draw_line(event);
	}
});

canvas.addEventListener('mousedown', function(event){
	// Remove welcome text the first time you click on the canvas
	if(init){
		clear_canvas();
		init = false;
	}
	x_0 = event.layerX;
	y_0 = event.layerY;

	mouse_pressed = true;
});

canvas.addEventListener('mouseup', function(){
	mouse_pressed = false;
	x_0, y_0 = (null, null);
});

canvas.addEventListener('mouseout', function(){
	mouse_pressed = false;
	x_0, y_0 = (null, null);
});


var ClassifyText = function (){

	// Hides any old error's
	$('.alert').hide();

	// Create border around canvas
	create_border(); 

	// Save canvas
	dataURL = canvas.toDataURL();

	$.ajax({
		url: 'http://www.google.com/12',
		method: 'GET'
	}) .done(function() {
		$('.results').show();
		$('#result').text('0900');
	})
	.fail(function(error) {
		console.log('my error', error, error.status, error.responseText, error.statusText);
		$('#error_status').text('Error ' + error.status);
		$('#error_test').text(error.statusText);
		$('.alert').show();
	});

	// console.log(dataURL);

}
