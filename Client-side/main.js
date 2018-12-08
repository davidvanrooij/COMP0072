window.onload = function() {

	// Add event listeners for desktop

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

		lock_scroll();
	});

	canvas.addEventListener('mouseup', function(){
		mouse_pressed = false;
		x_0, y_0 = (null, null);

		release_scroll();
	});

	canvas.addEventListener('mouseout', function(){
		mouse_pressed = false;
		x_0, y_0 = (null, null);

		release_scroll();
	});


	// Add event listeners for mobile

	canvas.addEventListener('touchmove', function(event){
		if(mouse_pressed){
			draw_line(event);
		}

		event.preventDefault();
	});

	canvas.addEventListener('touchstart', function(event){
		// Remove welcome text the first time you click on the canvas
		if(init){
			clear_canvas();
			init = false;
		}
		x_0 = event.layerX;
		y_0 = event.layerY;

		mouse_pressed = true;

		event.preventDefault();

		lock_scroll();
	});

	canvas.addEventListener('touchend', function(event){
		mouse_pressed = false;
		x_0, y_0 = (null, null);

		event.preventDefault();

		release_scroll();
	});

	canvas.addEventListener('touchcancel', function(event){
		return true;

		mouse_pressed = false;
		x_0, y_0 = (null, null);

		event.preventDefault();

		release_scroll();
	});

}

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

	// Hides any old error's
	$('.alert').hide();

	// Hide image string
	$('#imageURL').text('');

	// Hide old results
	$('.results').hide();
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

	// var cursor_speed = (Math.sqrt((x_1 - x_0)*(x_1 - x_0) + (y_1- y_0)*(y_1 - y_0)) + 1) * -0.3;
	// console.log(cursor_speed);

	ctx.beginPath();
	ctx.moveTo(x_0,y_0);
	ctx.lineTo(x_1,y_1);
	ctx.lineWidth = 10; // + cursor_speed;
	ctx.lineJoin = 'round';
	ctx.closePath();
	ctx.stroke();


	// Save curser coordinates as last point
	x_0 = x_1;
	y_0 = y_1;
}


function lock_scroll(){
	return true; 
	$('html, body').css('overflow', 'hidden');
}

function release_scroll(){
	return true;
	$('html, body').css('overflow', 'auto');
}


show_welcome_text();



var ClassifyText = function(){

	// Show loading icon
	$('.spinner').css('display', 'inline-flex');
	$('.label').hide();

	// Hides any old error's
	$('.alert').hide();

	// Hide former results
	$('.results').hide();

	// Create border around canvas
	create_border(); 

	// Save canvas
	canvas_url = canvas.toDataURL();

	$('#imageURL').text(canvas_url);

	console.log(canvas_url);

	$.ajax({
		url: 'https://bau-test-api.herokuapp.com/image',
		// url: 'http://localhost:5000/image',
		method: 'POST',
		crossDomain: true,
		data: { 
			imgBase64: canvas_url
		}
	}) .done(function(result) {

		result = result.join(' ')

		console.log(result);

		// Show results
		$('.results').show();
		$('#result').text(result);

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');

	})
	.fail(function(error) {

		// Show errors
		console.log('my error', error, error.status, error.responseText, error.statusText);
		$('#error_status').text('Error ' + error.status + ' (' + error.statusText + ')');
		$('#error_text').text(error.responseText);
		$('.alert').show();

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');
	});

}
