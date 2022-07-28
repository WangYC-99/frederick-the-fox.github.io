# Docker入门笔记

> by WangYC
>
> Sep.24 2021

## 第一部分 本地docker配置

### Docker简介

度娘：

![image-20210924085314233](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924085314233.png)

Wikipedia：

![image-20210924085450832](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924085450832.png)

人话：用户可以建立自己的系统镜像（images），来定制拥有适合自己的开发环境的“微型系统”。

### 初识Docker

经过我的简单实用，抽象总结docker为有git体验的虚拟机。

简单来讲就是我认为你可以吧docker理解为一个非常简单的虚拟机，只不过简单到没有图形化界面。

那为什么说是有git体验呢？

是因为docker有类似于github的社区，**docker hub**。

在docker hub上，用户开源自己定制的系统镜像，提供给他人下载。

![image-20210924094044639](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924094044639.png)

### Docker使用

#### push远程images到本地

利用简单的` docker pull env_name:tag_name` 命令即可将线上开源的docker images存储至本地。

```
docker pull nvcr.io/nvidia/l4t-base:r32.6.1
```

#### 查看本地images

```
docker images
```

![image-20210924095847651](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924095847651.png)

#### 运行image

命令格式` docker run -itd env_name:tag_name`

其中options `-itd` 意思:

-i:交互式运行。

-t:为容器重新分配一个伪终端进行输入。

-d:后台运行，即新run一个image不会把之前正在运行的image给kill掉。

```
docker run -it nvcr.io/nvidia/l4t-base:r32.6.1
```

#### 在x86环境下运行arm架构系统

方法：安装qume

MacOS:使用home brew安装

```
brew install qemu-user-static
```

Linux：能不能apt安装自己去查一下吧hhh

## 第二部分 体验升级：VScode + docker

利用VScode的extension“Remote-Containers"可以实现在vscode中管理资源程序

![image-20210924140816843](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924140816843.png)

step1:首先在本地运行docker的镜像，随后点击左侧工具栏的远程资源管理器

![image-20210924140941380](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924140941380.png)

step2:找到你要调试的image，右键进行连接

![image-20210924141500249](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924141500249.png)

step3:这时vscode左下角提醒你你已经连接到了远程container容器上

![image-20210924141643880](/Users/yc_wang/Library/Application Support/typora-user-images/image-20210924141643880.png)

选择你想要的文件夹就可以开始工作了。

## 第三部分 image构建上传



## 第四部分 拓展开发

docker还有可以不pull下来直接连接远程docker开发的功能，有志者可以去体验开发

