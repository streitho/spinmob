The source code for spinmob is constantly improving. If you find a bug in one of the official distributions, it may have already been fixed. This page outlines how to install the latest version of spinmob using an SVN client.

## Windows ##

  * Download and install [TortoiseSVN](http://tortoisesvn.tigris.org/)
  * Create a folder named "spinmob" in your [site-packages](SitePackages.md) directory.
  * Right click this new spinmob folder, select "SVN Checkout..."
  * In the box labeled "URL of repository" enter http://spinmob.googlecode.com/svn/trunk
  * Click OK.

All done. Any time you want to update, simply right-click this folder and select "SVN Update".

## Linux & Mac ##

  * Find your [site-packages](SitePackages.md) directory.
  * In the terminal window use the command
```
svn checkout http://spinmob.googlecode.com/svn/trunk /path/to/site-packages/spinmob
```
  * Then whenever you want to update spinmob to the latest version, use
```
svn update /path/to/site-packages/spinmob
```
  * It is possible to turn these commands into clickable scripts or aliased terminal commands. To make it an easy terminal command (e.g. spinmob-update), add the following line to ".profile" in your home directory:
```
alias spinmob-update="svn update /path/to/site-packages/spinmob"
```

  * I don't know what is the best SVN client. I remember at one point I used a clunky one in linux, but it worked.