#!/bin/bash

curdir=`pwd`
codedir=../Tools/CoreNLPServer

cd $codedir
java -Xmx2g -jar CoreNLPServer.jar

cd $curdir
