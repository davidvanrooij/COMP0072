window.onload = function() {

	// Add event listeners for desktop

	canvas.addEventListener('mousemove', function(event){
		if(mouse_pressed){
			draw_line(event.layerX, event.layerY);
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

	canvas.addEventListener('touchstart', function(event){

		var rect = canvas.getBoundingClientRect();

		// Remove welcome text the first time you click on the canvas
		if(init){
			clear_canvas();
			init = false;
		}

		x_0 = event.touches[0].clientX - rect.left;
		y_0 = event.touches[0].clientY - rect.top;

		mouse_pressed = true;

		lock_scroll();
	});

	canvas.addEventListener('touchmove', function(event){

		event.preventDefault();

		var rect = canvas.getBoundingClientRect();

		x_1 = event.touches[0].clientX - rect.left;
		y_1 = event.touches[0].clientY - rect.top;

		if(mouse_pressed){
			draw_line(x_1, y_1);
		}
	});

	canvas.addEventListener('touchend', function(event){

		mouse_pressed = false;
		x_0, y_0 = (null, null);

		release_scroll();
	});

	canvas.addEventListener('touchcancel', function(event){
		event.preventDefault();

		mouse_pressed = false;
		x_0, y_0 = (null, null);

		release_scroll();
	});

}

var ctx = document.getElementById('canvas').getContext('2d');

var init = true;
var mouse_pressed = false;
var x_0, y_0 = (0, 0);
var welcome_text = undefined;
var empty_canvas = undefined;


// Function to show welcome text
function show_welcome_text() {
	ctx.font = '20px Noto Sans';
	ctx.fillText('Start drawing symbols here!', 100, 150-10);
	welcome_text = true; 
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

	empty_canvas = true;
	welcome_text = false;
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
function draw_line(x_1,y_1) {
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


	empty_canvas = false;
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

	// Canvas with welcome text
	if(welcome_text){	
		$('#error_status').text('Don\'t be shy!');
		$('#error_text').text('Go and draw some numbers!');
		$('.alert').show();

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');

		return;
	}

	// Empty canvas
	if(empty_canvas){	
		$('#error_status').text('Are you trying to cheat?');
		$('#error_text').text('we cannot classify anything on an empty canvas!');
		$('.alert').show();

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');

		return;
	}

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

		console.log(result);

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');


		if(result.length == 0){
			$('#error_status').text('This is awkward...');
			$('#error_text').text('We were unable to classify anything');
			$('.alert').show();

			// Set icons back to default state
			$('.spinner').hide();
			$('.label').css('display', 'inline-flex');

			return;
		}

		result = result.join(' ');

		// Show results
		$('.results').show();
		$('#result').text(result);

	})
	.fail(function(error) {

		let error_text = error.responseJSON ? error.responseJSON.text : error.responseText

		// Show errors
		console.log('my error', error, error.status, error.responseText, error.statusText);
		$('#error_status').text('Error ' + error.status + ' (' + error.statusText + ')');
		$('#error_text').text(error_text);
		$('.alert').show();

		// Set icons back to default state
		$('.spinner').hide();
		$('.label').css('display', 'inline-flex');
	});

}
