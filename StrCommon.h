#ifndef _STRCOMMONH_
#define _STRCOMMONH_
#include <string>
#include <vector>
#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <time.h>

static int gettimeofday(struct timeval *tp, void *tzp)
{
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;

	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds * 1000;

	return (0);
}
static void usleep(int64_t us) {
	int64_t s = us / 1000;
	Sleep(s);
}
#else
#include <sys/time.h>
#endif

using namespace cv;


static int64_t gtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	int64_t re = ((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec;
	return re;
}

static void splitStr(const std::string& inputStr, const std::string &key, std::vector<std::string>& outStrVec) {
	if (inputStr == "") {
		return;
	}
	int pos = inputStr.find(key);
	int oldpos = 0;
	if (pos > 0) {
		std::string tmp = inputStr.substr(0, pos);
		outStrVec.push_back(tmp);
	}
	while (1) {
		if (pos < 0) {
			break;
		}
		oldpos = pos;
		int newpos = inputStr.find(key, pos + key.length());
		std::string tmp = inputStr.substr(pos + key.length(), newpos - pos - key.length());
		outStrVec.push_back(tmp);
		pos = newpos;
	}
	int tmplen = 0;
	if (outStrVec.size() > 0) {
		tmplen = outStrVec.at(outStrVec.size() - 1).length();
	}
	if (oldpos + tmplen < inputStr.length() - 1) {
		std::string tmp = inputStr.substr(oldpos + key.length());
		outStrVec.push_back(tmp);
	}
}

static std::string trim(std::string &s) {
	if (s.empty()) {
		return s;
	}

	s.erase(0, s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

static int toInt(const std::string &in){
	int re = 0;
	sscanf(in.c_str(), "%d", &re);
	return re;
}
static float toFloat(const std::string &in) {
	float re = 0;
	sscanf(in.c_str(), "%f", &re);
	return re;
}
static std::string toStr(float in) {
	char chr[20] = { 0 };
	sprintf(chr, "%f", in);
	std::string re(chr);
	return re;
}
static std::string toStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%d", in);
	std::string re(chr);
	return re;
}
static std::string to4dStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%04d", in);
	std::string re(chr);
	return re;
}
static std::string to5dStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%05d", in);
	std::string re(chr);
	return re;
}
static std::string to6dStr(int in){
	char chr[20] = {0};
	sprintf(chr, "%06d", in);
	std::string re(chr);
	return re;
}
#endif
