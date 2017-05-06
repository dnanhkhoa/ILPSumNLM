#!/bin/bash

curdir=`pwd`
codedir=../Tools/VnNLPServer

cd $codedir
java -mx500m -jar VnNLPServer.jar

cd $curdir
