#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define MAX_PATH 2000

void renameFilesInFolder(const char *folderPath, const char *extension)
{
    DIR *dir;
    struct dirent *entry;
    int count = 1;

    dir = opendir(folderPath);

    if (dir == NULL)
    {
        perror("Error opening directory");
        exit(EXIT_FAILURE);
    }

    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG)
        {
            char oldName[MAX_PATH];
            char newName[MAX_PATH];

            // Extract the original file extension
            const char *fileExtension = strrchr(entry->d_name, '.');
            if (fileExtension == NULL)
            {
                fprintf(stderr, "Error: File %s has no extension.\n", entry->d_name);
                exit(EXIT_FAILURE);
            }

            snprintf(oldName, sizeof(oldName), "%s/%s", folderPath, entry->d_name);
            snprintf(newName, sizeof(newName), "%s/%d.%s", folderPath, count, extension);

            if (rename(oldName, newName) != 0)
            {
                perror("Error renaming file");
                exit(EXIT_FAILURE);
            }

            count++;
        }
    }

    closedir(dir);
}

int main()
{
    char filename[] = "classnames.txt";


    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char folderPath[MAX_PATH];
    char extension[] = "jpeg"; // Set the file extension to 'jpeg'

    // Read folder names from the file and process each one
    while (fgets(folderPath, sizeof(folderPath), file) != NULL)
    {
        // Remove newline character if present
        size_t len = strlen(folderPath);
        if (len > 0 && folderPath[len - 1] == '\n')
        {
            folderPath[len - 1] = '\0';
        }

	    if (len > 0 && folderPath[len - 2] == '\r')
        {
            folderPath[len - 2] = '\0';
        }

        // Process the folder
        renameFilesInFolder(folderPath, extension);

        printf("Files in %s renamed numerically with the extension .%s.\n", folderPath, extension);
    }

    fclose(file);

    return 0;
}
