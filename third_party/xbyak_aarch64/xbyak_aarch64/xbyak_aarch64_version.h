static const int majorVersion = 1;
static const int minorVersion = 3;
static const int patchVersion = 0;
static int getVersion() { return (majorVersion << 16) + (minorVersion << 8) + patchVersion; }
static const char *getVersionString() { return "1.3.0"; }
