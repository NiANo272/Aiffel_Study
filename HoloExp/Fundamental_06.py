import copy

myword = 'cat'
print(myword.upper()) #모든 알파벳을 대문자로 (소문자는 lower)
print("cat".upper())

var = 4
print(id(var), id(4)) #4의 아이디 값과 var의 아이디 값은 같다

#얕은 복사
mylist = [1,2,3]
var = mylist
mylist.append(4) #var과 mylist모두에 동일한 연산이 적용됨
print(mylist)
print(var)
print(id(mylist), id(var)) #아이디 값이 같다.

#깊은 복사
varD = copy.deepcopy(mylist)
mylist.append(5)
print(mylist)
print(var)
print(varD)
print(id(mylist), id(varD)) #아이디 값이 다르다.

print("===========================")
#클래스
class Car:
    #속성
    color = 'blue'
    category = 'sports car'
    #메서드
    def drive(self):
        print("I am driving")

    def accel(self, speed_up, current_speed=10):
        self.speed_up = speed_up
        self.current_speed = current_speed + speed_up
        print("speed up :{}, driving at {}".format(speed_up, current_speed) )

#클래스 생성
mycar = Car()
print("차의 색깔은 {} 입니다.".format(mycar.color)) #차 색깔
mycar.drive()
mycar.accel(5)