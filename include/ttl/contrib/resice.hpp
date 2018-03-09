#pragma once

#include <cmath>

static inline int reseni_rovnic (double *a,double *x,double *y,long n,long m,long as, long* av)
/* funkce provadi reseni maticove rovnice A.X=Y
   lze ji tedy bez problemu pouzit na vypocet inverzni matice
   matice A je plna

   A(n,n),X(n,m),Y(n,m)
   as - rozhodovaci konstanta
   as=1 - pivot se hleda jen v pripade, ze A(i,i)=0
   as=2 - pivot se hleda pokazde
   av=n-element pivot array

   testovano 24.7.1996 pomoci programu test_fin_res.c v /u/jk/TESTY/METODY
   procedura dava stejne vysledky jako ldl,ldlkon,ldlblok,congrad_sky,congrad_comp

   **** otestovano ****
   */
{
  assert(a);
  assert(x);
  assert(y);
  assert(n);
  assert(m);
  assert(av);

  long    i,j,k,ac,acr,acc{},aca,aca1,acx,acy,acy1,aci,acj;
  double  s,g;

  /*************************************************************************/
  /*  nastaveni hodnot vektoru, ktery udava poradi jednotlivych neznamych  */
  /*************************************************************************/
  for (i=0;i<n;i++){
    av[i]=i;
  }

  for (i=0;i<n-1;i++){
    acr=i;  acc=i;
    if (as==1){
      /******************************************/
      /*  pivot se hleda jen pokud je A(i,i)=0  */
      /******************************************/
      if (fabs(a[i*n+i])<1.0e-5){
        /*  vyber pivota  */
        s=0.0;
        /*  smycka pres radky  */
        for (j=i;j<n;j++){
          aca=j*n+i;
          /*  smycka pres sloupce  */
          for (k=i;k<n;k++){
            if (s<fabs(a[aca])){
              s=fabs(a[aca]);  acr=j;  acc=k;
            }
            aca++;
          }
        }
        if (s==0.0){
          return 1;
        }
      }
    }
    if (as==2){
      /****************************/
      /*  pivot se hleda pokazde  */
      /****************************/
      s=0.0;
      /*  smycka pres radky  */
      for (j=i;j<n;j++){
        aca=j*n+i;
        /*  smycka pres sloupce  */
        for (k=i;k<n;k++){
          if (s<fabs(a[aca])){
            s=fabs(a[aca]);  acr=j;  acc=k;
          }
          aca++;
        }
      }
      if (s<1.0e-15){
        return 1;
      }
    }

    /******************/
    /*  vymena radku  */
    /******************/
    if (acr!=i){
      aca=i*n+i;  aca1=acr*n+i;
      for (j=i;j<n;j++){
        s=a[aca];
        a[aca]=a[aca1];
        a[aca1]=s;
        aca++;  aca1++;
      }
      acy=i*m;  acy1=acr*m;
      for (j=0;j<m;j++){
        s=y[acy];
        y[acy]=y[acy1];
        y[acy1]=s;
        acy++;  acy1++;
      }
    }
    /********************/
    /*  vymena sloupcu  */
    /********************/
    if (acc!=i){
      ac=av[i];
      av[i]=av[acc];
      av[acc]=ac;

      aca=i;  aca1=acc;
      for (j=0;j<n;j++){
        s=a[aca];
        a[aca]=a[aca1];
        a[aca1]=s;
        aca+=n;  aca1+=n;
      }
    }
    /***************/
    /*  eliminace  */
    /***************/

    for (j=i+1;j<n;j++){
      acj=j*n+i;  aci=i*n+i;
      s=a[acj]/a[aci];
      /*  modifikace matice A  */
      for (k=i;k<n;k++){
        a[acj]-=s*a[aci];
        acj++;  aci++;
      }
      acj=j*m;  aci=i*m;
      /*  modifikace matice pravych stran Y  */
      for (k=0;k<m;k++){
        y[acj]-=s*y[aci];
        acj++;  aci++;
      }
    }
  }

  /*****************/
  /*  zpetny chod  */
  /*****************/

  for (i=n-1;i>-1;i--){
    g=a[i*n+i];  acx=i*m;
    for (j=0;j<m;j++){
      s=0.0;  aca=i*n+i+1;  acy=(i+1)*m+j;
      for (k=i+1;k<n;k++){
        s+=a[aca]*x[acy];
        aca++;  acy+=m;
      }
      x[acx]=(y[acx]-s)/g;
      acx++;
    }
  }

  /***********************************/
  /*  prerovnani do puvodniho stavu  */
  /***********************************/
  for (i=0;i<n;i++){
    if (av[i]!=i){
      for (j=i;j<n;j++){
        if (av[j]==i){
          acc=j;  break;
        }
      }

      ac=av[i];
      av[i]=av[acc];
      av[acc]=ac;

      aca=i*m;  aca1=acc*m;
      for (j=0;j<m;j++){
        s=x[aca];
        x[aca]=x[aca1];
        x[aca1]=s;
        aca++;  aca1++;
      }
    }
  }

  return 0;
}
