<?php
$data = $_GET;
$python = "/Users/nitintokas/anaconda/bin/python";
if($data['x']==1){
	$file = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/get_screenshots1.py";
}
elseif ($data['x']==2) {
	$file = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/get_screenshots2.py";
}
elseif ($data['x']==3) {
	$file = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/get_screenshots3.py";
}
else{
	die("0");
}
$info = shell_exec("$python $file 2>&1");
//var_dump($data);
?>