# LeetCode 剑指

> by WangYC
>
> @NWPU changan Mar.30th- 2022

## 栈与队列

### [1. 剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

#### code

```cpp
class CQueue {
public:
    stack<int> s1;
    stack<int> s2;
    CQueue() {
    }
    
    void appendTail(int value) {
        while(!s1.empty())
        {
            s2.push(s1.top());
            s1.pop();
        }
        s1.push(value);
        while(!s2.empty())
        {
            s1.push(s2.top());
            s2.pop();
        }
    }
    
    int deleteHead() {
        if (s1.empty())
        {
            return -1;
        }
        else
        {
            int temp = s1.top();
            s1.pop();
            return temp;
        }
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */
```

#### 分析

easy题 注意的就是cpp的stack只有top()没有front()或者back()，添加用的push()即可

### [2. 剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

#### code

```cpp
class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {

    }

    stack<int> num;
    stack<int> min_current;

    void push(int x) {
        num.push(x);
        if (min_current.empty()) min_current.push(x);
        else 
        {
            min_current.push(x < min_current.top() ? x : min_current.top());
        }
    }
    
    void pop() {
        num.pop();
        min_current.pop();
    }
    
    int top() {
        return num.top();
    }
    
    int min() {
        return min_current.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

#### 分析

直接用了两个stack分别存储数据和状态。但是感觉不是非常通用的方法。

结果看到题解后，发现思路和我差不多哈哈，只不过题解的思路是第二个栈就不用每次都保存了，每当出现要新加入的值小于或者等于最小值的时候，就在辅助栈中再次插入，保证了最大化节省空间。