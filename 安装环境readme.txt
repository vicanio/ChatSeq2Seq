sudo service mysql start
sudo service mysql stop
sudo service mysql status
https://www.cnblogs.com/hupeng1234/p/7003543.html


1. vi /etc/mysql/mysql.conf.d/mysqld.cnf 
2.update user set host='%' where user='root' and host='localhost';
flush privileges;
select host,user from user;

3.python3.5升级python3.6
https://blog.csdn.net/qq_40965177/article/details/83500817

apt-get install python3-pip

pip3 install --upgrade pip


4.安装git
dpkg --configure -a
apt-get update
apt install git

5.python3.6 下提示No module named "apt_pkg"的解决方法
http://www.fantansy.cn/index.php/python/311.html



6.tensorflow
https://blog.csdn.net/algerwang/article/details/82802750


7.anaconda
https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh

https://blog.csdn.net/qq_34201858/article/details/85273350