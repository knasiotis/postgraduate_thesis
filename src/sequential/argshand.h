/*******************************************************************************/
#ifndef __ARGSHAND__
#define __ARGSHAND__

/*----------------------------------------------------------------------------*/
char       *GetArg  (const char *, int, char **);
bool        ExistArg(const char *, int, char **);

/*----------------------------------------------------------------------------*//*----------------------------------------------------------------------------*/
#endif /*__ARGSHAND__*/

/*----------------------------------------------------------------------------*/
/* args_hand.cc -- Manipulation of the input arguments. gse. 1997.            */
/*----------------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
/*----------------------------------------------------------------------------*/
/* Look for the string 'str_to_find' in 'arg'.                                */
/* If it is found, return the next string in arg.                             */
/* In other case return NULL.                                                 */
/*----------------------------------------------------------------------------*/
char *GetArg(const char *str_to_find, int narg, char **arg) {
    int i;                              
    for(i=0;i<narg;i++)                
        if(strstr(arg[i],str_to_find)) 
            return arg[i+1];          
    return NULL;           
}


/*----------------------------------------------------------------------------*/
/* Return 1 if the strind 'str_to_find' is in the command line 'arg'          */
/*----------------------------------------------------------------------------*/
bool ExistArg(const char *str_to_find, int narg, char **arg) {
  int i;
  for(i=0;i<narg;i++)
    if(strstr(arg[i],str_to_find)) return 1;
  return false;
}

/*----------------------------------------------------------------------------*//*----------------------------------------------------------------------------*/
