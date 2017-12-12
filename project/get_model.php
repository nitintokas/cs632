<?php
$data = $_GET;
$python = "/Users/nitintokas/anaconda/bin/python";
if($data['x']==1){
	$file = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/model.py";
}
elseif($data['x']==2){
	$file = "/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/predict.py";
}
$info = shell_exec("$python $file");
if($data['x']==1){
echo $info;
}
elseif($data['x']==2){
$info = preg_replace('!\s+!', ' ', trim(str_replace(array("[","]"), "", $info)));
$array = explode(" ", $info);
//var_dump($array);
	echo "<h3>Result</h3>";
	echo "<br><br>";
	echo "Group 1: ".round($array[0]*100,2)."%";
	echo "<br>";
	echo "Group 2: ".round($array[1]*100,2)."%";
	echo "<br>";
	echo "Group 3: ".round($array[2]*100,2)."%";
}
?>