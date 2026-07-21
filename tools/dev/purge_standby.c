// Purge the Windows standby (file cache) list so disk-bound benchmarks start cold.
// Equivalent to Sysinternals RAMMap "Empty Standby List". Requires admin.
#include <windows.h>
#include <stdio.h>

typedef enum _SYSTEM_MEMORY_LIST_COMMAND {
    MemoryCaptureAccessedBits,
    MemoryCaptureAndResetAccessedBits,
    MemoryEmptyWorkingSets,
    MemoryFlushModifiedList,
    MemoryPurgeStandbyList,
    MemoryPurgeLowPriorityStandbyList,
    MemoryCommandMax
} SYSTEM_MEMORY_LIST_COMMAND;

#define SystemMemoryListInformation 0x50

typedef LONG (WINAPI *NtSetSystemInformation_t)(INT, PVOID, ULONG);

static int enable_priv(const char *name) {
    HANDLE tok;
    TOKEN_PRIVILEGES tp;
    LUID luid;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return 0;
    if (!LookupPrivilegeValueA(NULL, name, &luid)) { CloseHandle(tok); return 0; }
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    BOOL ok = AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
    DWORD err = GetLastError();
    CloseHandle(tok);
    return ok && err == ERROR_SUCCESS;
}

int main(void) {
    if (!enable_priv("SeProfileSingleProcessPrivilege")) {
        fprintf(stderr, "ERR: cannot enable SeProfileSingleProcessPrivilege (run as admin)\n");
        return 2;
    }
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    NtSetSystemInformation_t NtSetSystemInformation =
        (NtSetSystemInformation_t)GetProcAddress(ntdll, "NtSetSystemInformation");
    if (!NtSetSystemInformation) { fprintf(stderr, "ERR: no NtSetSystemInformation\n"); return 2; }
    int cmd = MemoryPurgeStandbyList;
    LONG st = NtSetSystemInformation(SystemMemoryListInformation, &cmd, sizeof(cmd));
    if (st != 0) { fprintf(stderr, "ERR: NtSetSystemInformation 0x%lx\n", (unsigned long)st); return 1; }
    printf("standby list purged\n");
    return 0;
}
