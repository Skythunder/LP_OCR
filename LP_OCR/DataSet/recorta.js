var gm = require('gm');
var csv = require('csv');
var fs = require('fs');
var parse = require('csv-parse');

// gm("img.png").crop(width, height, x, y)

var parser = parse({columns:true, delimiter: ';'}, function(err, data){
  data.forEach(function(value, index){
  	console.log(value.image);
	gm(value.image)
		.crop(value.width,value.height,value.x,value.y)
		.write('positivos/' + index + '_' + value.plate + '.jpg', function(err) {
				if (!err) console.log('ok');
			});
    });
});

fs.createReadStream(__dirname+'/groundtruth.csv').pipe(parser);
