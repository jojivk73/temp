
#include <iostream>
#include <string>

#define ABCD

//--------------------------------------------------------
// template to cover data
//--------------------------------------------------------
template <bool VER1=true> struct CType {
   int a;
   float b;
};

//--------------------------------------------------------
// 1.Specialization for Class/struct based on template argument
//   if one class/struct has more data or fns.
//--------------------------------------------------------
template <> struct CType<true> {
  int a;
  float b;
  std::string c; // Additional data only for VER1=true
                 //Can do simlar for fn members
  int d;
};
//--------------------------------------------------------
//2. Use namespace for each version
//--------------------------------------------------------
namespace V1 {
   typedef CType<true> CT;
   static std::ostream &mout=std::cout;
   
   //3. Use fn templates for difference in arg type
   template <bool tp> inline int getValue(CType<tp> &ct)
   {
     return ct.d*7;
   }
   inline void doSomething(int x) {} // does nothing here
   inline void setD(CT &ct, int val) {  ct.d=val; } 
   inline void setC(CT &ct, const char *st) { ct.c= st; }
   inline void showC(CT & ct) { mout<<ct.c<<std::endl; }
   void whereAreYou() {mout<<" In V1...!"<<std::endl; }
}
//--------------------------------------------------------
// Second namespace for new version
//--------------------------------------------------------
namespace V2 {
   typedef CType<false> CT;
   static std::ostream &mout=std::cerr;
   template <bool tp> inline int getValue(CType<tp> &ct) { return 0; }// Nothing return zero
   inline void doSomething(int x) {
       mout<<"Here I am..!"<<x<<std::endl;
   } 

   // 4. Use Empty inline fns to cover code only in one case.
   //    That is this version of lib does not have the operation/fn
   inline void setD(CT &ct, int val) {}           //Empty
   inline void setC(CT &ct, const char *st) { }   //Empty
   inline void showC(CT & ct) { }                 //Empty
   void whereAreYou() {mout<<" In V2...!"<<std::endl; }
}
//--------------------------------------------------------
// 5. Single ifdef
//--------------------------------------------------------
#ifdef ABCD
  using namespace V1;
#else
  using namespace V2;
#endif

//--------------------------------------------------------
// Main uses the enabled version
//--------------------------------------------------------
int main()
{
  CT ct;
  whereAreYou();
  setC(ct, "Hello");
  showC(ct);
  setD(ct, 125);
  mout<<"d*7 ="<<getValue(ct)<<std::endl;
  doSomething(153);
}
