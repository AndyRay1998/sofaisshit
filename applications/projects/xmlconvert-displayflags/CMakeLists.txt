cmake_minimum_required(VERSION 3.12)
project(xmlconvert-displayflags)

find_package(TinyXML REQUIRED)

add_executable(${PROJECT_NAME} xmlvisitor.h xmlvisitor.cpp main.cpp)
target_link_libraries(${PROJECT_NAME} TinyXML Sofa.Core)
