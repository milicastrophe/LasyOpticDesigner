import numpy as np
from tensorflow import Variable
import tensorflow as tf
from scipy.optimize import root,fsolve
# import tensorflow as tf

class Tools:
    @staticmethod
    def afirmDerivable(fun,x,acu=0.0000000001):
        af = abs(abs(fun(x)-fun(x+acu))-abs(fun(x)-fun(x-acu)))
        if af < abs(fun(x)-fun(x+acu)):
            return True
        else:
            return False
 
    @staticmethod
    def getDerivative(fun, x, dx=0.0000000001):
        """单值函数求导"""
        return (fun(x+dx)-fun(x))/dx
        

    @staticmethod
    def getIntegral(fun, a, b, dx=0.0001):
        temp = 0
        for i in range(int(1/dx)):
            temp += fun(a+(b-a)*i*dx)
        return temp

    @staticmethod
    def get_part(fun, val_index, val, acu=0.00000001):
        val = val.astype(np.float)
        newval = val.copy()      
        newval[val_index] += acu
        return (fun(newval)-fun(val))/acu

    @staticmethod
    def get_grid(fun, val):
        temp = []
        for i in range(val.size):
            temp.append(Tools.get_part(fun, i, val))
        return np.array(temp)
    
    @staticmethod
    def get_nodical(fun1,fun2,point=0):
        """
        At a plant for get nodical with fun1 and fun2
        
        """
        return Tools.get_nodical_borders(fun1,fun2,point,'l'),Tools.get_nodical_borders(fun1,fun2,point,'r')
    
    @staticmethod
    def afirm_nodical_sphere_line(point,R,rp,direct):
        base1 = np.linalg.norm((point-rp),ord=2)
        base2 = np.linalg.norm(direct,ord=2)
        afirm_r = ((base1*base2)**2-(point-rp).dot(direct))**0.5/base2
        if afirm_r<=R:
            return True
        else:
            return
   
    @staticmethod
    def get_nodical_sphere_ray(rp,direct,c,position):
        R = 1/c
        r = abs(R)
        center = np.array([0,0,position+R])
        line = center-rp

        base_dir = np.linalg.norm(direct,2)
        base_line = np.linalg.norm(line,2)
        
        cos_rp = direct.dot(line)/(base_dir*base_line)
        sin_rp = (1-cos_rp**2)**0.5
        if sin_rp == 0:
            z1 = np.sqrt(np.square([R,rp[0],rp[1]]))+R+position
            z2 = -np.sqrt(np.square([R,rp[0],rp[1]]))+R+position
            af1 = (z1-rp[2])*direct[2]
            af2 = (z2-rp[2])*direct[2]
            if af2>0 and af1<0:
                return np.array([rp[0],rp[1],z1])
            elif af2<0 and af1>0:
                return np.array([rp[0],rp[1],z2])
            elif af2>0 and af1>0:
                if abs(z2-rp[2])>abs(z1-rp[2]):
                    return np.array([rp[0],rp[1],z1])
                else:
                    return np.array([rp[0],rp[1],z2])

        sin_nodical = base_line*sin_rp/r
        if base_line*2**0.5>r:
            cos_nodical = -(1-sin_nodical**2)**0.5
        else:
            cos_nodical = (1-sin_nodical**2)**0.5
        
        sin_center = cos_nodical*sin_rp+cos_rp*sin_nodical
        base_ray = sin_center*r/sin_rp
        nodical = rp+direct*base_ray/base_dir
        return nodical

    @staticmethod
    def get_nodical_standard_ray(c,position,rp,direct,K=-np.exp(2)):
        """
        get nodical interact by ray and standard surface
        direct[0] !=0 ,direct[1] !=0 ,direct[2] !=0
        """      
        standard = lambda r : c*(r[0]**2+r[1]**2)-(r[2]-position)*(1+(1-(1+K)*c**2*(r[0]**2+r[1]**2))**0.5)
        line_fun1 = lambda r : (r[0]-rp[0])/direct[0]-(r[1]-rp[1])/direct[1]
        line_fun2 = lambda r : (r[0]-rp[0])/direct[0]-(r[2]-rp[2])/direct[2] 
        def inner(r):
            return np.array([
                standard(r),
                line_fun1(r),
                line_fun2(r)
            ])
        return fsolve(inner,[0,0,0])
        
    @staticmethod
    def get_nodical_borders(fun1, fun2,point,lor):    
        """
        获得交点:
        """
        point = float(point)
        va= point
        if fun2(va)>fun1(va):
            x,x_ = Tools.fun_range(fun1,fun2,point,lor)
        elif fun1(va)>fun2(va):
            x,x_ = Tools.fun_range(fun2,fun1,point,lor)
        init_diff = abs(fun2(x)-fun1(x))
        if init_diff > 0.0000000001:
            init_diff=0.0000000001
        while abs(fun1(x)-fun2(x)) >init_diff*0.0001:
            if (fun1((x+x_)/2)-fun2((x+x_)/2))*(fun1(x)-fun2(x))>0:
                x = (x+x_)/2
            elif (fun1((x+x_)/2)-fun2((x+x_)/2))*(fun1(x)-fun2(x))<0:
                x_ = (x+x_)/2
            else:
                return (x+x_)/2
        return x if abs(fun1(x)-fun2(x))<abs(fun1(x_)-fun2(x_)) else x_
 
   
    @staticmethod
    def fun_range(fun1,fun2,point,l_or_r):#fun2 must > fun1 l_or_r: 'l','r'
        """
        the sun function of get_nodical_borders
        fun1:the biger one of callable 
        fun2:the smaller one of callable
        point: the init point of solve
        l_or_r:'l' ,'r' the sign of the point 
        """
        point=float(point)
        x = point
        lr= 0.05
        x_= point
        loss = fun2(x)-fun1(x)
        while loss >0:   
            x_ = x  
            if l_or_r=='l':
                x -= loss*lr 
            else:
                x += loss*lr
            loss = fun2(x)-fun1(x) 
            
        return x,x_ #fun2(x_)-fun1(x_)>0

    @staticmethod
    def getPartIntergal(fun,val_index,fy1,fy2):
        """
        fy1 is the 
        """
        for i in range(10000):
            pass

    @staticmethod
    def FFtdem2(fun,w):
        def inner(z):
            return fun(z)*np.exp(-2j*np.pi(np.dot(z.T,w)))
        # outter = Tools.getPartIntergal(inner,0)
        return inner
    
    @staticmethod
    def get_center(point_array):
        return np.mean(point_array,axis=0)

    @staticmethod
    def tf_get_center(point_array):
        return tf.reduce_mean(point_array,axis=0)

    @staticmethod
    def get_max(fun,a,b):
        n = 0
        init_root = (a+b)*0.5
        inner_init = lambda x:fun(x)-n
        s = root(inner_init,[init_root])
        while s['success']==True:
            inner = lambda x:fun(x)-n
            n_=n
            n+=10
            s = root(inner,[init_root])
        
        while n-n_>0.000001:
            temp = (n+n_)/2
            new_inner_init = lambda x:fun(x)-temp
            s = root(new_inner_init,[init_root])
            if s['success']==True:
                n_=temp
            else:
                n=temp
        final_root = root(lambda x: fun(x)-n_,[init_root])['x'][0]
        return [final_root,n_]



class Line:
    def __init__(self,rp,direct):
        # if isinstance(rp,tf.Tensor) and isinstance(rp,tf.Tensor):
        self.rp = rp
        self.direct = direct

def expression_fun(val):
    """
    val: tf.Variable
    """
    return tf.sqrt(tf.add(tf.square(val[0]),tf.square(val[1])))

class Linear(tf.keras.layers.Layer):
    """
    线性回归模块
    """
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)	
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )	        
        self.b = self.add_weight(
            shape=(self.units,),	                            
            initializer='random_normal',	                             
            trainable=True
        )	

    def call(self, inputs):	
        return tf.matmul(inputs, self.w) + self.b 

class GetNodicalLineFaceFreedom:
    """
    find the nodical which create by interact beteen the ray and Surface
    """
    def __init__(self,rp,direct,face_fun,acu):
        """
        rp: reference point for the Ray start
        direct: the direct of Ray
        face_fun: the expression of face
        acu : the accuracy of computation
        """
        self.rp = rp
        self.direct=direct
        self.face_fun = face_fun
        self.acu = acu
  
    def get_range(self):
        temp = self.rp
        while self.face_fun(temp[:2])==None:
            temp*=0.5
            
        if self.face_fun(temp[:2]):
            afirm_value = self.face_fun(temp[:2])
            step = self.direct*(afirm_value-self.rp[2])
            point =temp+step
            start = temp
            mun = (self.face_fun(point[:2])-point[2])* (self.face_fun(start[:2])-start[2])
            while mun>0:
                point += step
                mun = (self.face_fun(point[:2])-point[2])* (self.face_fun(start[:2])-start[2])
            return start,point

    def get_nodical(self):
        start,tail = self.get_range()
        while self.norm(start,tail) > self.acu:
            
            newpoint = (start+tail)/2
            if (self.face_fun(newpoint[:2])-newpoint[2])*(self.face_fun(start[:2])-start[2])>0:
                start = newpoint
            elif (self.face_fun(newpoint[:2])-newpoint[2])*(self.face_fun(start[:2])-start[2])<0:
                tail = newpoint
            else:
                return newpoint      
        return start

    def norm(self,point1,point2):
        dis = point1-point2
        return np.sum(np.square(dis))


class Function:
    def __init__(self, _f):
        self.fun = _f

    def value(self, val):
        return self.fun(val)

    def part(self, var_index, val):
        """
        偏导数值
        """
        a = self.fun(val)
        b = a + 1
        i = 0
        e = 2 ** 10 - 1
        e1 = 2 ** 10
        while 10 ** (-6) < e < e1 or i > -6:
            e1 = e
            a = b
            val_ = list(val)
            val_[var_index] += 10 ** i
            m = self.fun(val_)
            n = self.fun(val)
            b = (m - n) / 10 ** i
            i -= 2
            e = abs(b - a)
        return a

    def part_2(self, x_index, y_index, val):
        """
        二阶导数
        """
        return self.__diff_(x_index).__diff_(y_index).value(val)

    def diff(self, val):
        """
        导数
        """
        a = self.fun(val)
        b = a + 1
        i = 0
        e = 2 ** 10 - 1
        e1 = 2 ** 10
        while 10 ** (-6) < e < e1 or i > -6:
            e1 = e
            a = b
            val_ = val + 10 ** i
            m = self.fun(val_)
            n = self.fun(val)
            b = (m - n) / 10 ** i
            i -= 2
            e = abs(b - a)
        return a

    def grad(self, val):
        """
        梯度
        """
        g = np.array(val).astype('float')
        for i in range(0, g.size):
            g[i] = self.part(i, val)
        return np.array(g)

    def __diff_(self, index):
        """
        偏导数
        """
        def diff_f(vals):
            vals_ = list(vals)
            vals_[index] = vals_[index] + 10 ** (-6)
            m = self.fun(vals_)
            n = self.fun(vals)
            return (m - n) / 10 ** (-6)
        return Function(diff_f)

    def hesse(self, val):
        """
        黑森
        """
        v = np.mat(val)
        G = np.mat(np.dot(v.T, v)).astype('float')
        for i in range(0, v.size):
            for j in range(0, v.size):
                p = self.part_2(i, j, val)
                G[i, j] = p
        return G

    def norm(self, val):
        """
        范数
        """
        s = 0
        for x in self.grad(val):
            s += x ** 2
        return np.sqrt(s)
    
    def get_max(self,a,b):
        pass

    def get_min(self,a,b):
        pass


class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


# 链表类
class Linklist:
    """
    建立链表模型
    进行链表操作
    """

    def __init__(self):
        """
        初始化链表,生成一个头节点,表示链表开始节点
        """
        self.head = Node(None)

    # 初始添加一组链表节点
    def init_list(self, list_):
        p = self.head  # ｐ为移动变量
        for i in list_:
            p.next = Node(i)
            p = p.next  # ｐ向后移动一个节点

    # 遍历链表
    def show(self):
        p = self.head.next  # 第一个有效节点
        while p is not None:
            print(p.data, end=' ')
            p = p.next
            print()  # 换行

    # 获取链表长度
    def get_length(self):
        n = 0
        p = self.head
        while p.next is not None:
            n += 1
            p = p.next
        return n

    # 判断链表是否为空
    def is_empty(self):
        if self.get_length() == 0:
            return True
        else:
            return False

    # 清空链表
    def clear(self):
        self.head.next = None

    #　尾部插入节点
    def append(self,data):
        node = Node(data)  #　生成新节点
        p = self.head
        while p.next is not None:
            p = p.next
            p.next = node

    #　选择位置插入节点
    def insert(self,index,data):
        if index < 0 or index > self.get_length():
            raise IndexError("index out of range")

        # 定义ｐ移动到插入位置的前一个
        p = self.head
        for i in range(index):
            p = p.next

        node = Node(data) #　生成节点

        node.next = p.next
        p.next = node

    
    def delete(self,data):
        p = self.head
        while p.next and p.next.data != data:
            p = p.next
        
        if p.next is None:
            raise ValueError("value is error")
        else:
            p.next = p.next.next


    def get_item(self,index):
        """
        get value of node
        """
        if index < 0 or index >= self.get_length():
            raise IndexError("index out of range")
        p = self.head.next
        for i in range(index):
            p = p.next
        return p.data



    
